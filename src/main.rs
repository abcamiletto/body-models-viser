use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model_data: PathBuf,
    #[arg(long)]
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output = body_models_viser::run_fixture(&args.model_data, &args.input)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
