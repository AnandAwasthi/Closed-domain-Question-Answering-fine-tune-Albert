
from transformers.examples.run_squad import RunSquad
import argparse

class TrainBalanceSheetQnA:
    def __init__(self, 
        model_type,
        model_name_or_path,
        output_dir,
        data_dir,
        train_file,
        predict_file
        ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.train_file = train_file
        self.predict_file = predict_file
       

    def train(self):
        run_squad = RunSquad(self.model_type, self.model_name_or_path, self.output_dir, self.data_dir, self.train_file, self.predict_file)

        run_squad.execute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    args = parser.parse_args()
    train_bs_qna = TrainBalanceSheetQnA(
                    model_type= args.model_type,
                    model_name_or_path = args.model_name_or_path,
                    output_dir = args.output_dir,
                    data_dir = args.data_dir,
                    train_file= args.train_file,
                    predict_file = args.predict_file
                    )
    
    train_bs_qna.train()

if __name__ == "__main__":
    main()











    