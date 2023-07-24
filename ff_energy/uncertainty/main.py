import pandas as pd
from dataloader import DataLoader
from uncertainty_quantifier import UncertaintyQuantifier
from plotter import Plotter
from latex_table_generator import LatexTableGenerator

def main():
    # Load and preprocess the data
    data_loader = DataLoader('data.csv')
    data = data_loader.load_data()

    # Calculate the uncertainty
    uncertainty_quantifier = UncertaintyQuantifier(data)
    uncertainty = uncertainty_quantifier.calculate_uncertainty()

    # Generate the plots
    plotter = Plotter(data, uncertainty)
    plotter.generate_plots()

    # Generate the Latex table
    latex_table_generator = LatexTableGenerator(data, uncertainty)
    latex_table_generator.generate_table()

if __name__ == "__main__":
    main()
