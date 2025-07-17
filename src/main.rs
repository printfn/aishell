use std::{env, fs, os::unix::process::ExitStatusExt, process::Stdio};

use async_openai::{
	Client,
	config::OpenAIConfig,
	types::{
		ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
		ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
		ChatCompletionRequestUserMessageArgs, ChatCompletionToolArgs, ChatCompletionToolType,
		CreateChatCompletionRequestArgs, FunctionObjectArgs,
	},
};
use eyre::bail;
use rustyline::{Config, Editor, error::ReadlineError, history::DefaultHistory};
use serde::Deserialize;
use serde_json::json;
use tokio::{process::Command, spawn};
use tracing::{trace, warn};

fn init_context() -> eyre::Result<Vec<ChatCompletionRequestMessage>> {
	Ok(vec![
		ChatCompletionRequestSystemMessageArgs::default()
			.content("Use the exec tool to run bash commands that corresponds to the user's request. These will be run in a child process, so e.g. 'cd' will not persist outside your command. Don't hesitate to run multiple tool commands. When you've received the results, add an explanation.")
			.build()?
			.into(),
	])
}

fn add_prompt(ctx: &mut Vec<ChatCompletionRequestMessage>, prompt: &str) -> eyre::Result<()> {
	ctx.push(
		ChatCompletionRequestUserMessageArgs::default()
			.content(prompt)
			.build()?
			.into(),
	);
	Ok(())
}

async fn ai(
	client: &Client<OpenAIConfig>,
	ctx: &mut Vec<ChatCompletionRequestMessage>,
) -> eyre::Result<String> {
	let request = CreateChatCompletionRequestArgs::default()
		.max_completion_tokens(4096u32)
		.model("gpt-4.1-mini")
		.messages(ctx.as_slice())
		.tools(vec![
			ChatCompletionToolArgs::default()
				.r#type(ChatCompletionToolType::Function)
				.function(
					FunctionObjectArgs::default()
						.name("exec")
						.description("execute a bash command")
						.parameters(json!({
							"type": "object",
								"properties": {
									"command": {
										"type": "string",
										"description": "The bash command to execute, e.g. `ping 127.0.0.1`",
									},
									"timeout": {
										"type": "number",
										"description": "Timeout in seconds for this bash command (default: 10 seconds)",
									},
									"dangerous": {
										"type": "boolean",
										"description": "Is this command potentially dangerous? This will require the user to manually accept the command before it will be executed.",
									},
								},
								"required": ["command"],
						}))
						.build()?,
				)
				.build()?,
		])
		.build()?;
	let response = client.chat().create(request).await?;
	if response.choices.is_empty() {
		bail!("empty response (`choices` is empty)");
	}
	let message = response.choices.into_iter().next().unwrap().message;
	trace!("< {message:?}");
	if let Some(tool_calls) = message.tool_calls {
		let mut handles = Vec::new();
		for call in tool_calls.iter() {
			if call.function.name != "exec" {
				bail!("unknown function {}", call.function.name);
			}
			let args: ExecArgs = serde_json::from_str(&call.function.arguments)?;
			handles.push((
				spawn(execute_bash(args.command, args.timeout, args.dangerous)),
				call.clone(),
			));
		}
		ctx.push(
			ChatCompletionRequestAssistantMessageArgs::default()
				.tool_calls(tool_calls)
				.build()?
				.into(),
		);
		for (handle, call) in handles {
			if let Ok(response_content) = handle.await {
				ctx.push(
					ChatCompletionRequestToolMessageArgs::default()
						.content(response_content.unwrap_or_else(|e| format!("error: {e}")))
						.tool_call_id(call.id)
						.build()?
						.into(),
				);
			}
		}
		return Box::pin(ai(client, ctx)).await;
	}
	let Some(content) = message.content else {
		bail!("no content in response message");
	};
	Ok(content)
}

#[derive(Deserialize)]
struct ExecArgs {
	command: String,
	timeout: Option<f64>,
	#[serde(default)]
	dangerous: bool,
}

async fn execute_bash(
	command: String,
	timeout: Option<f64>,
	dangerous: bool,
) -> eyre::Result<String> {
	if dangerous {
		bail!("not executing dangerous command {command:?}");
	}
	if let Some(timeout) = timeout {
		warn!("timeout: {timeout} seconds");
	}
	eprintln!(":: {command}");
	let mut cmd = Command::new("/usr/bin/env");
	cmd.arg("bash")
		.arg("-c")
		.arg(command)
		.stdout(Stdio::piped())
		.stderr(Stdio::piped());
	let child = cmd.spawn()?;
	let result = child.wait_with_output().await?;
	print!("{}", String::from_utf8_lossy(&result.stdout));
	eprint!("{}", String::from_utf8_lossy(&result.stderr));
	Ok(serde_json::to_string(&json!({
		"exit_code": result.status.into_raw(),
		"stdout": String::from_utf8_lossy(&result.stdout),
		"stderr": String::from_utf8_lossy(&result.stderr),
	}))?)
}

async fn handle_command(
	client: &Client<OpenAIConfig>,
	ctx: &mut Vec<ChatCompletionRequestMessage>,
	rl: &mut Editor<(), DefaultHistory>,
) -> eyre::Result<()> {
	let pwd = env::current_dir()?;
	let prompt = format!("{} ", pwd.display());
	let line = rl.readline(&prompt)?;
	if line.trim() == "clear" {
		*ctx = init_context()?;
		eprintln!("context cleared");
		return Ok(());
	}
	add_prompt(ctx, &line)?;
	let response = ai(client, ctx).await?;
	eprintln!("{response}");
	Ok(())
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
	tracing_subscriber::fmt::init();
	let Some(home_dir) = env::home_dir() else {
		bail!("couldn't determine home directory");
	};
	let api_key_path = home_dir.join(".config/aishell/openai-api-key");
	fs::create_dir_all(&api_key_path.parent().unwrap())?;
	let api_key = match fs::read_to_string(&api_key_path) {
		Ok(key) => key.trim().to_string(),
		Err(e) => {
			bail!(
				"failed to open OpenAI API key in {}: {e}",
				api_key_path.display()
			);
		}
	};
	let client = Client::with_config(OpenAIConfig::new().with_api_key(api_key));
	let mut ctx = init_context()?;
	let mut rl = rustyline::Editor::with_config(
		Config::builder()
			.max_history_size(1_000_000)?
			.history_ignore_space(true)
			.auto_add_history(true)
			.build(),
	)?;
	let history_path = home_dir.join(".local/state/aishell/history");
	fs::create_dir_all(&history_path.parent().unwrap())?;
	let _ = rl.load_history(&history_path);
	loop {
		let outcome = handle_command(&client, &mut ctx, &mut rl).await;
		rl.save_history(&history_path)?;
		match outcome {
			Ok(()) => (),
			Err(e) => {
				if let Some(ReadlineError::Eof | ReadlineError::Interrupted) =
					e.downcast_ref::<ReadlineError>()
				{
					return Ok(());
				}
				eprintln!("error: {e}");
				continue;
			}
		}
	}
}
