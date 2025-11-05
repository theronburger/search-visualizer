import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Button Test
    """)
    return


@app.cell
def _(mo):
    # Create a simple button with on_click to increment
    test_button = mo.ui.button(
        label="Click Me",
        value=0,
        on_click=lambda count: count + 1
    )
    test_button
    return (test_button,)


@app.cell
def _(mo, test_button):
    # This cell should react when button is clicked
    if test_button.value > 0:
        mo.md(f"✓ Button clicked {test_button.value} times!")
    else:
        mo.md("⚠ Button not clicked yet")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell
def _(mo):
    # Test with run_button pattern
    run_button = mo.ui.run_button()

    mo.vstack([
        mo.md("### Using run_button:"),
        run_button
    ])
    return (run_button,)


@app.cell
def _(mo, run_button):
    # Check run_button value
    if run_button.value:
        mo.md(f"✓ Run button activated! Value: {run_button.value}")
    else:
        mo.md("⚠ Run button not activated")
    return


if __name__ == "__main__":
    app.run()
