import parametrization
import plot_exp
import argparse

start = 2000
stop = -1
step = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-p", "--path", help="path to data", required=True)
    parser.add_argument("-l", "--label", help="data label", required=True)
    args = vars(parser.parse_args())

    path = args["path"]
    label = args["label"]
    dataclass = parametrization.fill_multirun_prob_dataclass(path)
    dataclass.slice_data(start, stop, step)
    df = dataclass.multirun_df
    #  plot energy, temperature, pressure
    plot_exp.plot_traj_data(df)
    #  plot properties
    plot_exp.plot_properties([df], [label], FONTSIZE=14)

    out = plot_exp.test_properties(df, label)
    test_df = plot_exp.pd.DataFrame([out])
    test_df.to_csv(f"out_data/{label}.csv", index=False)
