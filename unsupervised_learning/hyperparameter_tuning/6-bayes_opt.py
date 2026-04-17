#!/usr/bin/env python3
"""Optimize a Keras model with Bayesian optimization using GPyOpt."""
import gc
import os

import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


MAX_TOTAL_ITERATIONS = 30
INITIAL_DESIGN_POINTS = 5
BO_ITERATIONS = MAX_TOTAL_ITERATIONS - INITIAL_DESIGN_POINTS
EPOCHS = 20
PATIENCE = 3
TRAIN_SAMPLES = 12000
VAL_SAMPLES = 2000
SEED = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "bayes_opt_checkpoints")
REPORT_PATH = os.path.join(BASE_DIR, "bayes_opt.txt")
PLOT_PATH = os.path.join(BASE_DIR, "bayes_opt_convergence.png")


def format_token(value):
    """Create compact checkpoint-safe tokens from floats and ints."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    token = f"{float(value):.2e}"
    return token.replace("+", "").replace("-", "m").replace(".", "p")


class FashionMnistBayesOpt:
    """Bayesian optimization runner for a simple Fashion-MNIST classifier."""

    def __init__(self):
        """Load data, define the search space, and prepare bookkeeping."""
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255.0

        self.X_train = x_train[:TRAIN_SAMPLES]
        self.Y_train = y_train[:TRAIN_SAMPLES]
        self.X_val = x_train[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES]
        self.Y_val = y_train[TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES]

        self.domain = [
            {
                "name": "log_learning_rate",
                "type": "continuous",
                "domain": (-4.0, -2.0),
            },
            {
                "name": "units_1",
                "type": "discrete",
                "domain": (64, 128, 192, 256),
            },
            {
                "name": "units_2",
                "type": "discrete",
                "domain": (32, 64, 96, 128),
            },
            {
                "name": "dropout_1",
                "type": "continuous",
                "domain": (0.0, 0.5),
            },
            {
                "name": "dropout_2",
                "type": "continuous",
                "domain": (0.0, 0.5),
            },
            {
                "name": "log_l2",
                "type": "continuous",
                "domain": (-6.0, -3.0),
            },
            {
                "name": "batch_size",
                "type": "discrete",
                "domain": (32, 64, 128, 256),
            },
        ]

        self.trials = []
        self.eval_count = 0

    def decode_params(self, x_row):
        """Map the GPyOpt vector into training hyperparameters."""
        return {
            "learning_rate": 10 ** float(x_row[0]),
            "units_1": int(x_row[1]),
            "units_2": int(x_row[2]),
            "dropout_1": float(x_row[3]),
            "dropout_2": float(x_row[4]),
            "l2_weight": 10 ** float(x_row[5]),
            "batch_size": int(x_row[6]),
        }

    def checkpoint_path(self, params):
        """Build a checkpoint filename containing the tuned hyperparameters."""
        filename = (
            f"trial_{self.eval_count:02d}"
            f"_lr_{format_token(params['learning_rate'])}"
            f"_u1_{params['units_1']}"
            f"_u2_{params['units_2']}"
            f"_d1_{format_token(params['dropout_1'])}"
            f"_d2_{format_token(params['dropout_2'])}"
            f"_l2_{format_token(params['l2_weight'])}"
            f"_bs_{params['batch_size']}.keras"
        )
        return os.path.join(CHECKPOINT_DIR, filename)

    def build_model(self, params):
        """Create and compile a dense classifier."""
        regularizer = tf.keras.regularizers.l2(params["l2_weight"])

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                params["units_1"],
                activation="relu",
                kernel_regularizer=regularizer,
            ),
            tf.keras.layers.Dropout(params["dropout_1"]),
            tf.keras.layers.Dense(
                params["units_2"],
                activation="relu",
                kernel_regularizer=regularizer,
            ),
            tf.keras.layers.Dropout(params["dropout_2"]),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=params["learning_rate"]
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def objective(self, x):
        """Train once and return the value minimized by GPyOpt."""
        params = self.decode_params(x[0])
        self.eval_count += 1
        checkpoint = self.checkpoint_path(params)

        model = self.build_model(params)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=PATIENCE,
                mode="max",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            self.X_train,
            self.Y_train,
            validation_data=(self.X_val, self.Y_val),
            epochs=EPOCHS,
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        val_accuracy = history.history["val_accuracy"]
        best_val_accuracy = float(np.max(val_accuracy))
        best_epoch = int(np.argmax(val_accuracy) + 1)
        objective_value = 1.0 - best_val_accuracy

        self.trials.append({
            "iteration": self.eval_count,
            "objective": objective_value,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "epochs_ran": len(val_accuracy),
            "checkpoint": checkpoint,
            "hyperparameters": params,
        })

        tf.keras.backend.clear_session()
        gc.collect()
        return np.array([[objective_value]])

    def write_report(self, optimizer):
        """Save the optimization report to bayes_opt.txt."""
        best_trial = min(self.trials, key=lambda trial: trial["objective"])

        with open(REPORT_PATH, "w", encoding="utf-8") as report:
            report.write("Bayesian Optimization Report\n")
            report.write("============================\n")
            report.write("Dataset: Fashion MNIST\n")
            report.write("Model: Dense neural network\n")
            report.write("Single metric optimized: validation accuracy\n")
            report.write(f"Total evaluations: {len(self.trials)}\n")
            report.write(f"Maximum allowed evaluations: {MAX_TOTAL_ITERATIONS}\n\n")

            report.write("Best Result\n")
            report.write("-----------\n")
            report.write(
                f"Best objective (1 - val_accuracy): "
                f"{best_trial['objective']:.6f}\n"
            )
            report.write(
                f"Best validation accuracy: "
                f"{best_trial['best_val_accuracy']:.6f}\n"
            )
            report.write(f"Best epoch: {best_trial['best_epoch']}\n")
            report.write(f"Checkpoint: {best_trial['checkpoint']}\n")
            report.write(f"Optimizer x_opt: {optimizer.x_opt}\n")
            report.write(f"Optimizer fx_opt: {optimizer.fx_opt}\n")
            report.write("Best hyperparameters:\n")
            for name, value in best_trial["hyperparameters"].items():
                report.write(f"  {name}: {value}\n")

            report.write("\nTrial History\n")
            report.write("-------------\n")
            for trial in self.trials:
                report.write(
                    f"Iteration {trial['iteration']:02d}: "
                    f"objective={trial['objective']:.6f}, "
                    f"val_accuracy={trial['best_val_accuracy']:.6f}, "
                    f"best_epoch={trial['best_epoch']}, "
                    f"epochs_ran={trial['epochs_ran']}, "
                    f"checkpoint={trial['checkpoint']}\n"
                )
                for name, value in trial["hyperparameters"].items():
                    report.write(f"  {name}: {value}\n")

    def plot_convergence(self):
        """Save a convergence plot of the optimization history."""
        objectives = np.array([trial["objective"] for trial in self.trials])
        best_so_far = np.minimum.accumulate(objectives)
        iterations = np.arange(1, len(self.trials) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(
            iterations,
            objectives,
            marker="o",
            label="Objective per evaluation",
        )
        plt.plot(
            iterations,
            best_so_far,
            linewidth=2,
            label="Best objective so far",
        )
        plt.xlabel("Evaluation")
        plt.ylabel("Objective (1 - best val_accuracy)")
        plt.title("GPyOpt Convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        plt.close()

    def run(self):
        """Execute Bayesian optimization and return the configured optimizer."""
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.objective,
            domain=self.domain,
            initial_design_numdata=INITIAL_DESIGN_POINTS,
            acquisition_type="EI",
            exact_feval=False,
            maximize=False,
        )
        optimizer.run_optimization(max_iter=BO_ITERATIONS)
        self.plot_convergence()
        self.write_report(optimizer)
        return optimizer


if __name__ == "__main__":
    runner = FashionMnistBayesOpt()
    optimizer = runner.run()

    best_trial = min(runner.trials, key=lambda trial: trial["objective"])
    print("Best validation accuracy:", best_trial["best_val_accuracy"])
    print("Best hyperparameters:", best_trial["hyperparameters"])
    print("Report saved to:", REPORT_PATH)
    print("Convergence plot saved to:", PLOT_PATH)
