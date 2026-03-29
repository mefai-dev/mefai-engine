"""MEFAI Engine CLI - command line interface."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="mefai-engine",
    help="MEFAI Engine - Institutional-grade AI trading engine for crypto perpetual futures",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """Start the trading engine in live mode."""
    from mefai_engine.config import load_config
    from mefai_engine.constants import EngineMode

    settings = load_config(config)
    settings.engine.mode = EngineMode.LIVE
    console.print("[bold green]Starting MEFAI Engine (LIVE)[/bold green]")
    console.print(f"Symbols: {settings.engine.symbols}")
    console.print(f"Exchanges: {[k for k, v in settings.exchanges.__dict__.items() if hasattr(v, 'enabled') and v.enabled]}")

    _start_server(config, 8080)


@app.command()
def paper(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Config file path"),
    port: int = typer.Option(8080, "--port", "-p"),
) -> None:
    """Start the trading engine in paper trading mode."""
    import os
    os.environ["MEFAI__ENGINE__MODE"] = "paper"
    console.print(f"[bold cyan]Starting MEFAI Engine (PAPER) on port {port}[/bold cyan]")
    _start_server(config, port)


@app.command()
def api(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port", "-p"),
) -> None:
    """Start API server only (no trading loop)."""
    console.print(f"[bold blue]Starting MEFAI Engine API on {host}:{port}[/bold blue]")
    _start_server(config, port, host)


def _start_server(config_path: str, port: int = 8080, host: str = "0.0.0.0") -> None:
    """Start the uvicorn server with the FastAPI app."""
    import os
    os.environ["MEFAI_CONFIG_PATH"] = config_path

    import uvicorn
    uvicorn.run(
        "mefai_engine.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


@app.command()
def backtest(
    config: str = typer.Option("configs/backtest.yaml", "--config", "-c"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s"),
    start: str = typer.Option("2024-01-01", "--start"),
    end: str = typer.Option("2025-01-01", "--end"),
) -> None:
    """Run a backtest."""
    console.print("[bold magenta]Running Backtest[/bold magenta]")
    console.print(f"Symbol: {symbol} | Period: {start} to {end}")


@app.command()
def train(
    model: str = typer.Option("all", "--model", "-m", help="Model to train: gradient_boost, transformer, rl, sentiment, all"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Train or retrain ML models."""
    console.print(f"[bold yellow]Training Model: {model}[/bold yellow]")


@app.command()
def status() -> None:
    """Show engine status and current positions."""
    table = Table(title="MEFAI Engine Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Engine", "Not Running")
    table.add_row("Database", "Unknown")
    table.add_row("Redis", "Unknown")
    table.add_row("Exchanges", "Not Connected")
    console.print(table)


@app.command("config")
def validate_config(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Validate configuration file."""
    try:
        from mefai_engine.config import load_config
        settings = load_config(config)
        console.print("[green]Configuration valid[/green]")
        console.print(f"Mode: {settings.engine.mode}")
        console.print(f"Symbols: {settings.engine.symbols}")
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from mefai_engine import __version__
    console.print(f"MEFAI Engine v{__version__}")


if __name__ == "__main__":
    app()
