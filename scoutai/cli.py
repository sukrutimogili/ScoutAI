"""
ScoutAI CLI entry point.

Full implementation in S14 (hiring-run command).
This stub ensures the package is importable and the entry-point is declared.
"""

import typer

app = typer.Typer(
    name="scoutai",
    help="ScoutAI — Evidence-driven AI hiring intelligence system.",
    add_completion=False,
)


@app.command("hiring-run")
def hiring_run(
    jd: str = typer.Option(..., "--jd", help="Path to job description file."),
    resumes: str = typer.Option(..., "--resumes", help="Path to directory of résumé files."),
    config: str = typer.Option("config.yaml", "--config", help="Path to config.yaml."),
) -> None:
    """Run the full hiring pipeline for a job description against a batch of résumés."""
    typer.echo("ScoutAI hiring pipeline — full implementation in S14.")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
