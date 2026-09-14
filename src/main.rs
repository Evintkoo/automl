//! AutoML - Main Entry Point
//!
//! A high-performance AutoML framework in Rust with CLI and server modes.

use clap::Parser;
use automl::cli::{Cli, Commands, cmd_train, cmd_predict, cmd_preprocess, cmd_benchmark, cmd_info, cmd_serve};
use tracing::{info, error};

fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("automl=info,automl=info,tower_http=info"));

    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());

    if is_tty {
        fmt()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false)
            .init();
    } else {
        // Structured JSON logging for non-interactive / production environments
        fmt()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .json()
            .init();
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let cli = Cli::parse();

    info!(version = env!("CARGO_PKG_VERSION"), "AutoML starting");

    let result = run(cli).await;

    if let Err(ref e) = result {
        error!(error = %e, "AutoML exited with error");
    }

    result
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Some(Commands::Train { data, target, model, task, cv_folds, output }) => {
            info!(data = %data.display(), target = %target, model = %model, "Starting training");
            cmd_train(&data, &target, &model, &task, cv_folds, output.as_deref())?;
        }
        Some(Commands::Predict { model, data, output }) => {
            info!(model = %model.display(), data = %data.display(), "Starting prediction");
            cmd_predict(&model, &data, output.as_deref())?;
        }
        Some(Commands::Preprocess { data, output, scaler, imputation }) => {
            info!(data = %data.display(), scaler = %scaler, imputation = %imputation, "Starting preprocessing");
            cmd_preprocess(&data, &output, &scaler, &imputation)?;
        }
        Some(Commands::Benchmark { data, target, task, cv_folds }) => {
            info!(data = %data.display(), target = %target, task = %task, "Starting benchmark");
            cmd_benchmark(&data, &target, &task, cv_folds)?;
        }
        Some(Commands::Info { data }) => {
            info!(data = %data.display(), "Inspecting dataset");
            cmd_info(&data)?;
        }
        Some(Commands::Serve { port, host }) => {
            info!(host = %host, port = port, "Starting server");
            cmd_serve(&host, port).await?;
        }
        None => {
            let host = std::env::var("API_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
            let port: u16 = std::env::var("API_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080);
            info!(host = %host, port = port, "No command specified, starting server with web UI");
            cmd_serve(&host, port).await?;
        }
    }

    info!("AutoML finished");
    Ok(())
}
