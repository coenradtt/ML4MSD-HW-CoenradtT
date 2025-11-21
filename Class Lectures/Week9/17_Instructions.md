# Updating Instructions

For this exercise, I build a separate UV environment inside the `Week9` folder. Here are the steps to update this and select the correct Kernel for the Jupyter Notebook.

1. Run `cd Week9` in the terminal
2. Run `uv sync` in the terminal
3. Activate Environment:<br>
Windows: Run `.\.venv\Scripts\activate`<br>
Mac: Run `source .venv/bin/activate`
4. Run `uv run ipython kernel install --user --name=ml4msd-mlip`
5. Restart VSCode (close it and reopen it)
6. Inside the `17_MLIPs.ipynb` click on "select Kernel" on the top right.
7. Click on "Select another Kernel"
8. Click on "Jupyter Kernel..."
9. Click on the "ml4msd-mlip" Kernel

## Alternative

If the above steps cause issues, you can also follow along in this [Google Colab Notebook](https://colab.research.google.com/drive/1_ByBimOZhDFWdwMkn9_KCzADVL7p4uxT).

However, only the first section works in that notebook.