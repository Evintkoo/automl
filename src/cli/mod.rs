//! AutoML CLI Module
//!
//! Command-line interface for training, prediction, and data processing.

use clap::{Parser, Subcommand};
use colored::*;
use polars::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn, debug, error};

use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy};
use crate::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

// ─── Styling helpers ───────────────────────────────────────────────────────────

const W: usize = 58; // box inner width

fn dim(s: &str) -> ColoredString   { s.truecolor(100, 100, 100) }
fn accent(s: &str) -> ColoredString { s.truecolor(120, 170, 255) }
fn muted(s: &str) -> ColoredString  { s.truecolor(140, 140, 140) }
fn ok(s: &str) -> ColoredString     { s.truecolor(100, 210, 120) }

fn line_box_top()    { println!("  {}", dim("┌─────────────────────────────────────────────────────────┐")); }
fn line_box_bottom() { println!("  {}", dim("└─────────────────────────────────────────────────────────┘")); }
fn line_box_sep()    { println!("  {}", dim("├─────────────────────────────────────────────────────────┤")); }

fn line_box(content: &str) {
    let visible_len = strip_ansi(content).len();
    let pad = if visible_len < W { W - visible_len } else { 0 };
    println!("  {}  {}{} {}", dim("│"), content, " ".repeat(pad), dim("│"));
}

fn line_box_center(content: &str) {
    let visible_len = strip_ansi(content).len();
    let total_pad = if visible_len < W { W - visible_len } else { 0 };
    let left = total_pad / 2;
    let right = total_pad - left;
    println!("  {}  {}{}{} {}", dim("│"), " ".repeat(left), content, " ".repeat(right), dim("│"));
}

fn line_box_empty() { line_box(""); }

fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' { in_escape = true; continue; }
        if in_escape { if c == 'm' { in_escape = false; } continue; }
        out.push(c);
    }
    out
}

fn kv(key: &str, val: &str) -> String {
    format!("{} {}", muted(key), val.white())
}

#[allow(dead_code)]
fn step_ok(msg: &str) {
    println!("  {} {}", ok("✓"), msg);
}

fn step_run(msg: &str) {
    print!("  {} {}... ", accent("›"), msg);
}

fn step_done(detail: &str) {
    println!("{} {}", ok("done"), dim(detail));
}

fn section(title: &str) {
    println!();
    println!("  {}", title.white().bold());
    println!("  {}", dim(&"─".repeat(56)));
}

fn wait_enter() {
    println!();
    println!("  {}", dim("press enter to continue"));
    let mut input = String::new();
    let _ = std::io::stdin().read_line(&mut input);
}

// ─── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "automl")]
#[command(author = "Evintkoo")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "High-performance AutoML framework in Rust")]
#[command(long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train a model on data
    Train {
        /// Input data file (CSV, JSON, or Parquet)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Model type (linear, logistic, decision_tree, random_forest)
        #[arg(short, long, default_value = "random_forest")]
        model: String,

        /// Task type (classification, regression)
        #[arg(long, default_value = "classification")]
        task: String,

        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        cv_folds: usize,

        /// Output model file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Make predictions using a trained model
    Predict {
        /// Trained model file
        #[arg(short, long)]
        model: PathBuf,

        /// Input data file
        #[arg(short, long)]
        data: PathBuf,

        /// Output predictions file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Preprocess data
    Preprocess {
        /// Input data file
        #[arg(short, long)]
        data: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Scaler type (none, standard, minmax, robust)
        #[arg(long, default_value = "standard")]
        scaler: String,

        /// Imputation strategy (drop, mean, median, mode)
        #[arg(long, default_value = "mean")]
        imputation: String,
    },

    /// Benchmark multiple models
    Benchmark {
        /// Input data file
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Task type (classification, regression)
        #[arg(long, default_value = "classification")]
        task: String,

        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        cv_folds: usize,
    },

    /// Show data information
    Info {
        /// Input data file
        #[arg(short, long)]
        data: PathBuf,
    },

    /// Start the web server
    Serve {
        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Server host
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
    },
}

// ─── Data loading ──────────────────────────────────────────────────────────────

pub fn load_data(path: &PathBuf) -> anyhow::Result<DataFrame> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    debug!(path = %path.display(), extension = ext, "Loading data file");

    let df = match ext {
        "csv" => CsvReadOptions::default()
            .with_infer_schema_length(Some(1000))
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.clone()))?
            .finish()?,
        "json" => JsonReader::new(std::fs::File::open(path)?)
            .finish()?,
        "parquet" => ParquetReader::new(std::fs::File::open(path)?)
            .finish()?,
        _ => anyhow::bail!("Unsupported file format: {}", ext),
    };

    info!(rows = df.height(), columns = df.width(), path = %path.display(), "Data loaded successfully");
    Ok(df)
}

// ─── Commands ──────────────────────────────────────────────────────────────────

pub fn cmd_train(
    data_path: &PathBuf,
    target: &str,
    model_type: &str,
    task_type: &str,
    _cv_folds: usize,
    _output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    section("Train");

    step_run("Loading data");
    let start = Instant::now();
    let df = load_data(data_path)?;
    step_done(&format!("{} rows × {} cols in {:?}", df.height(), df.width(), start.elapsed()));

    let task = match task_type {
        "classification" => TaskType::BinaryClassification,
        "multiclass" => TaskType::MultiClassification,
        "regression" => TaskType::Regression,
        _ => anyhow::bail!("Invalid task type: {}", task_type),
    };

    let model = match model_type {
        "linear" | "linear_regression" => ModelType::LinearRegression,
        "logistic" | "logistic_regression" => ModelType::LogisticRegression,
        "decision_tree" => ModelType::DecisionTree,
        "random_forest" => ModelType::RandomForest,
        _ => anyhow::bail!("Invalid model type: {}", model_type),
    };

    let config = TrainingConfig::new(task.clone(), target)
        .with_model(model);

    step_run(&format!("Training {}", model_type.cyan()));
    let start = Instant::now();
    let mut engine = TrainEngine::new(config);
    engine.fit(&df)?;
    let elapsed = start.elapsed();
    step_done(&format!("{:?}", elapsed));

    let metrics = engine.metrics().cloned().unwrap_or_default();
    let (metric_name, score) = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification =>
            ("Accuracy", metrics.accuracy.unwrap_or(0.0)),
        TaskType::Regression | TaskType::TimeSeries =>
            ("R²", metrics.r2.unwrap_or(0.0)),
        TaskType::Clustering =>
            ("Clusters", 0.0),
    };

    info!(
        model = model_type,
        task = task_type,
        metric = metric_name,
        score = score,
        training_time_secs = elapsed.as_secs_f64(),
        "Training completed"
    );

    println!();
    println!("  {:<16} {}", muted(metric_name), format!("{:.4}", score).white().bold());
    println!("  {:<16} {}", muted("Time"), format!("{:.3}s", metrics.training_time_secs).white());
    println!();

    Ok(())
}

pub fn cmd_predict(
    _model_path: &PathBuf,
    _data_path: &PathBuf,
    _output: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    section("Predict");
    println!("  {}", "Model persistence not yet implemented".yellow());
    println!();
    Ok(())
}

pub fn cmd_preprocess(
    data_path: &PathBuf,
    output_path: &PathBuf,
    scaler: &str,
    imputation: &str,
) -> anyhow::Result<()> {
    section("Preprocess");

    step_run("Loading data");
    let df = load_data(data_path)?;
    step_done(&format!("{} rows × {} cols", df.height(), df.width()));

    let scaler_type = match scaler {
        "standard" => ScalerType::Standard,
        "minmax" => ScalerType::MinMax,
        "robust" => ScalerType::Robust,
        _ => ScalerType::None,
    };

    let imputation_strategy = match imputation {
        "mean" => ImputeStrategy::Mean,
        "median" => ImputeStrategy::Median,
        "mode" => ImputeStrategy::MostFrequent,
        _ => ImputeStrategy::Drop,
    };

    let config = PreprocessingConfig::default()
        .with_scaler(scaler_type)
        .with_numeric_impute(imputation_strategy);

    step_run("Processing");
    let start = Instant::now();
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df)?;
    step_done(&format!("{:?}", start.elapsed()));

    step_run(&format!("Saving → {}", output_path.display()));
    let mut file = std::fs::File::create(output_path)?;
    CsvWriter::new(&mut file).finish(&mut processed.clone())?;
    step_done(&format!("{} rows × {} cols", processed.height(), processed.width()));

    println!();
    Ok(())
}

pub fn cmd_benchmark(
    data_path: &PathBuf,
    target: &str,
    task_type: &str,
    _cv_folds: usize,
) -> anyhow::Result<()> {
    section("Benchmark");

    step_run("Loading data");
    let df = load_data(data_path)?;
    step_done(&format!("{} rows × {} cols", df.height(), df.width()));

    let task = match task_type {
        "classification" => TaskType::BinaryClassification,
        "multiclass" => TaskType::MultiClassification,
        "regression" => TaskType::Regression,
        _ => anyhow::bail!("Invalid task type: {}", task_type),
    };

    let models: Vec<(&str, ModelType)> = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification => vec![
            ("Logistic Regression", ModelType::LogisticRegression),
            ("Decision Tree", ModelType::DecisionTree),
            ("Random Forest", ModelType::RandomForest),
            ("Gradient Boosting", ModelType::GradientBoosting),
            ("KNN", ModelType::KNN),
            ("Naive Bayes", ModelType::NaiveBayes),
            ("SVM", ModelType::SVM),
        ],
        TaskType::Regression | TaskType::TimeSeries => vec![
            ("Linear Regression", ModelType::LinearRegression),
            ("Decision Tree", ModelType::DecisionTree),
            ("Random Forest", ModelType::RandomForest),
            ("Gradient Boosting", ModelType::GradientBoosting),
            ("KNN", ModelType::KNN),
            ("SVM", ModelType::SVM),
        ],
        TaskType::Clustering => vec![
            ("KMeans", ModelType::KMeans),
            ("DBSCAN", ModelType::DBSCAN),
        ],
    };

    let metric_name = match task {
        TaskType::BinaryClassification | TaskType::MultiClassification => "Accuracy",
        TaskType::Regression | TaskType::TimeSeries => "R²",
        TaskType::Clustering => "Inertia",
    };

    println!();
    println!("  {:<24} {:>10} {:>10}", muted("Model"), muted(metric_name), muted("Time"));
    println!("  {}", dim(&"─".repeat(46)));

    let mut results: Vec<(String, f64, std::time::Duration)> = Vec::new();

    for (name, model_type) in models {
        let config = TrainingConfig::new(task.clone(), target)
            .with_model(model_type);

        let mut engine = TrainEngine::new(config);
        let start = Instant::now();

        match engine.fit(&df) {
            Ok(_) => {
                let elapsed = start.elapsed();
                let metrics = engine.metrics().cloned().unwrap_or_default();
                let score = match task {
                    TaskType::BinaryClassification | TaskType::MultiClassification =>
                        metrics.accuracy.unwrap_or(0.0),
                    TaskType::Regression | TaskType::TimeSeries =>
                        metrics.r2.unwrap_or(0.0),
                    TaskType::Clustering => 0.0,
                };

                println!("  {:<24} {:>10.4} {:>10.2?}", name, score, elapsed);
                info!(model = name, score = score, elapsed_secs = elapsed.as_secs_f64(), "Benchmark model completed");
                results.push((name.to_string(), score, elapsed));
            }
            Err(e) => {
                println!("  {:<24} {:>10}", name, format!("err: {}", e).red());
                error!(model = name, error = %e, "Benchmark model failed");
            }
        }
    }

    println!("  {}", dim(&"─".repeat(46)));

    if let Some((name, score, _)) = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        println!();
        println!("  {} {} {} {:.4}",
            ok("best"),
            name.white().bold(),
            muted(&format!("{}:", metric_name)),
            score
        );
        info!(best_model = %name, metric = metric_name, score = score, "Benchmark completed");
    }

    println!();
    Ok(())
}

pub fn cmd_info(data_path: &PathBuf) -> anyhow::Result<()> {
    section("Data Info");

    let df = load_data(data_path)?;

    println!("  {:<12} {}", muted("File"), data_path.display());
    println!("  {:<12} {}", muted("Rows"), df.height());
    println!("  {:<12} {}", muted("Columns"), df.width());
    println!("  {:<12} {:.2} MB", muted("Memory"), df.estimated_size() as f64 / 1024.0 / 1024.0);
    println!();

    println!("  {:<20} {:<12} {:>6} {:>8}", muted("Column"), muted("Type"), muted("Nulls"), muted("Unique"));
    println!("  {}", dim(&"─".repeat(50)));

    for col in df.get_columns() {
        println!(
            "  {:<20} {:<12} {:>6} {:>8}",
            col.name(),
            format!("{:?}", col.dtype()).truecolor(140, 140, 140),
            col.null_count(),
            col.n_unique().unwrap_or(0)
        );
    }

    println!();
    Ok(())
}

// ─── Serve ─────────────────────────────────────────────────────────────────────

pub async fn cmd_serve(host: &str, port: u16) -> anyhow::Result<()> {
    use crate::server::{run_server, ServerConfig};

    // Use localhost for display URLs when binding to all interfaces
    let display_host = if host == "0.0.0.0" || host == "::" { "localhost" } else { host };

    println!();
    line_box_top();
    line_box_empty();
    line_box_center(&format!("{}", "AutoML".white().bold()));
    line_box_center(&format!("{}", dim(&format!("v{}", env!("CARGO_PKG_VERSION")))));
    line_box_empty();
    line_box_sep();
    line_box_empty();
    line_box(&kv("Web UI ", &format!("http://{}:{}", display_host, port)));
    line_box(&kv("API    ", &format!("http://{}:{}/api", display_host, port)));
    line_box(&kv("Health ", &format!("http://{}:{}/api/health", display_host, port)));
    line_box_empty();
    line_box_sep();
    line_box_empty();
    line_box_center(&format!("{}", dim("ctrl+c to stop")));
    line_box_empty();
    line_box_bottom();
    println!();

    let config = ServerConfig {
        host: host.to_string(),
        port,
        ..Default::default()
    };

    // Auto-open browser
    let url = format!("http://{}:{}", display_host, port);
    #[cfg(target_os = "macos")]
    { let _ = std::process::Command::new("open").arg(&url).spawn(); }
    #[cfg(target_os = "linux")]
    { let _ = std::process::Command::new("xdg-open").arg(&url).spawn(); }
    #[cfg(target_os = "windows")]
    { let _ = std::process::Command::new("cmd").args(["/c", "start", &url]).spawn(); }

    run_server(config).await
}

// ─── Interactive mode ──────────────────────────────────────────────────────────

fn print_banner() {
    println!();
    println!();
    println!("       {}", "╻┏━  ┏━┓╻  ┏━┓┏━┓┏━┓╻     ┏━┓╻".truecolor(120, 170, 255));
    println!("       {}", "┣┻┓  ┃ ┃┃  ┃ ┃┗━┓┣━┫┃  ╺━╸┣━┫┃".truecolor(100, 150, 240));
    println!("       {}", "╹ ╹  ┗━┛┗━╸┗━┛┗━┛╹ ╹┗━╸   ╹ ╹╹".truecolor(80, 130, 220));
    println!();
    println!("       {}", dim(&format!("AutoML Framework  ·  v{}  ·  rust", env!("CARGO_PKG_VERSION"))));
    println!();
}

fn show_system_info() {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    section("System");

    println!("  {:<12} {}", muted("OS"), System::name().unwrap_or_else(|| "unknown".into()));
    println!("  {:<12} {}", muted("Arch"), std::env::consts::ARCH);
    println!("  {:<12} {}", muted("CPUs"), sys.cpus().len());
    println!("  {:<12} {:.1} / {:.1} GB", muted("Memory"),
        sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
    );
    println!("  {:<12} v{}", muted("AutoML"), env!("CARGO_PKG_VERSION"));
    println!();
}

fn default_host() -> String {
    std::env::var("API_HOST").unwrap_or_else(|_| "0.0.0.0".to_string())
}

fn default_port() -> u16 {
    std::env::var("API_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080)
}

fn show_help() {
    section("Commands");

    let cmds: &[(&str, &str)] = &[
        ("automl", "Interactive launcher (default)"),
        ("automl serve", "Start web UI + API server"),
        ("automl serve -p 3000", "Serve on custom port"),
        ("automl train -d data.csv -t col", "Train a model"),
        ("automl benchmark -d data.csv -t col", "Compare all models"),
        ("automl preprocess -d in.csv -o out.csv", "Preprocess data"),
        ("automl info -d data.csv", "Inspect a dataset"),
    ];

    for (cmd, desc) in cmds {
        println!("  {:<44} {}", cmd.white(), muted(desc));
    }

    section("Endpoints (default)");

    let host = default_host();
    let port = default_port();

    let endpoints: &[(&str, &str)] = &[
        ("", "Web dashboard"),
        ("/api", "REST API root"),
        ("/api/health", "Health check"),
        ("/api/system/status", "System status"),
    ];

    for (path, desc) in endpoints {
        let url = format!("http://{}:{}{}", host, port, path);
        println!("  {:<44} {}", url.truecolor(120, 170, 255), muted(desc));
    }

    println!();
}

pub async fn cmd_interactive() -> anyhow::Result<()> {
    use dialoguer::{Select, theme::ColorfulTheme};

    if !std::io::IsTerminal::is_terminal(&std::io::stdin())
        || !std::io::IsTerminal::is_terminal(&std::io::stderr())
    {
        warn!("No interactive terminal detected, showing usage help");
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        cmd.print_help()?;
        println!();
        return Ok(());
    }

    print_banner();

    let theme = ColorfulTheme {
        active_item_prefix: dialoguer::console::style("  ›".to_string()).for_stderr().cyan(),
        active_item_style: dialoguer::console::Style::new().for_stderr().white().bold(),
        inactive_item_prefix: dialoguer::console::style("   ".to_string()).for_stderr(),
        inactive_item_style: dialoguer::console::Style::new().for_stderr().color256(245),
        prompt_prefix: dialoguer::console::style("  ?".to_string()).for_stderr().color256(111),
        prompt_style: dialoguer::console::Style::new().for_stderr().white().bold(),
        ..ColorfulTheme::default()
    };

    let host = default_host();
    let port = default_port();

    loop {
        let items = &[
            format!("Start Server          web ui + rest api on {}:{}", host, port),
            "System Info           hardware & runtime details".to_string(),
            "Help                  commands & endpoints".to_string(),
            "Exit".to_string(),
        ];

        println!();
        let sel = Select::with_theme(&theme)
            .with_prompt("What would you like to do")
            .items(items)
            .default(0)
            .interact_opt()?;

        match sel {
            Some(0) => {
                info!(host = %host, port = port, "Starting server from interactive mode");
                cmd_serve(&host, port).await?;
                break;
            }
            Some(1) => {
                show_system_info();
                wait_enter();
            }
            Some(2) => {
                show_help();
                wait_enter();
            }
            Some(3) | None => {
                println!();
                println!("  {}", dim("goodbye"));
                println!();
                info!("User exited interactive mode");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
