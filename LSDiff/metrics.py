import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, f1_score, roc_auc_score,
    classification_report, r2_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from imblearn.metrics import geometric_mean_score


class DatasetEvaluator:
    def __init__(self, real_data, gen_data, col_names, path_dir="checkpoints/"):
        """
        Initialize the evaluator.
        :param real_data: np.array containing the real data (2D or 3D).
        :param gen_data: np.array containing the generated data (2D or 3D).
        :param col_names: list of column names in order.
        :param path_dir: save directory.
        """
        self.real_data = real_data
        self.gen_data = gen_data
        self.col_names = col_names
        self.path = path_dir
        os.makedirs(path_dir, exist_ok=True)
        self.output_file = "evaluation.txt"
        self.event_idx = col_names.index('event_code')
        self.col_indices = [i for i in range(len(col_names))]
        self.continuous_indices = [i for i, col in enumerate(col_names) if col != 'event_code' and col != 'temps_sec']
        self.num_vars = len(self.continuous_indices)
        self.num_patients = self.real_data.shape[0] if self.real_data.ndim == 3 else "N/A"
        self.is_3d = (self.real_data.ndim == 3)

    def _write_and_print(self, text, file):
        """
        Write to file and console.
        """
        print(text)
        file.write(text + "\n")

    def _get_column_vector(self, data, idx):
        """
        Extract a variable as a 1D vector (2D or 3D).
        """
        if self.is_3d:
            return data[:, :, idx].flatten()
        else:
            return data[:, idx].flatten()

    def _calculate_mmd(self, x, y, gamma=1.0):
        """
        Calculate an approximation of the Maximum Mean Discrepancy (MMD).
        """
        if len(x) > 5000:
            idx = np.random.choice(len(x), 5000, replace=False)
            x, y = x[idx], y[idx]
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        xx = np.exp(-gamma * (x - x.T)**2)
        yy = np.exp(-gamma * (y - y.T)**2)
        xy = np.exp(-gamma * (x - y.T)**2)
        return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

    def _calculate_multiclass_auc_pr(self, y_true, y_pred_labels, classes):
        """
        Calculate the macro AUC-PR by encoding One-vs-Rest and ignoring empty classes.
        """
        if len(classes) <= 1:
            return 0.0
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_bin = label_binarize(y_pred_labels, classes=classes)
        
        if y_true_bin.shape[1] == 1:
            if np.sum(y_true_bin) == 0:
                return 0.0
            precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
            return auc(recall, precision)
            
        auc_pr_list = []
        for i in range(len(classes)):
            if np.sum(y_true_bin[:, i]) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            auc_pr_list.append(auc(recall, precision))
            
        return np.mean(auc_pr_list) if len(auc_pr_list) > 0 else 0.0

    def _calculate_fsg(self, real_events, gen_events, classes_eval, weights=None, epsilon=1e-6):
        """
        Calculate the FSG (F-score, Weighted Geometric Mean).
        """
        if len(classes_eval) == 0:
            return 0.0

        f1_scores = f1_score(
            real_events, 
            gen_events, 
            labels=classes_eval, 
            average=None, 
            zero_division=0
        )

        if weights is None:
            w = np.ones(len(classes_eval), dtype=float)
        elif isinstance(weights, dict):
            w = np.array([weights.get(c, 1.0) for c in classes_eval], dtype=float)
        else:
            w = np.array(weights, dtype=float)

        sum_w = np.sum(w)
        if sum_w <= 0:
            return 0.0

        f1_clipped = np.clip(f1_scores, epsilon, 1.0)

        log_fsg = np.sum(w * np.log(f1_clipped)) / sum_w
        return float(np.exp(log_fsg))
    def _calculate_discrete_multiclass_auc_roc(self, real_events, gen_events, classes_eval):
            """
            Calculate the macro and weighted AUC-ROC One-vs-Rest on discrete categorical labels.
            """
            if len(classes_eval) <= 1:
                return 0.0, 0.0
    
            y_true_bin = label_binarize(real_events, classes=classes_eval)
            y_pred_bin = label_binarize(gen_events, classes=classes_eval)
    
            roc_scores = []
            weights = []
    
            for i, cls in enumerate(classes_eval):
                positives = np.sum(y_true_bin[:, i])
                negatives = len(y_true_bin) - positives
    
                if positives == 0 or negatives == 0:
                    continue
    
                score = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
                roc_scores.append(score)
                weights.append(positives)
    
            roc_scores = np.array(roc_scores)
            weights = np.array(weights)
    
            macro_auc_roc = np.mean(roc_scores) if len(roc_scores) > 0 else 0.0
            weighted_auc_roc = np.average(roc_scores, weights=weights) if len(roc_scores) > 0 else 0.0
    
            return float(macro_auc_roc), float(weighted_auc_roc)

    def evaluate_continuous_variables(self, file=None):
        """
        Calculate all regression and distributional metrics.
        Returns:
          - "by_variable" : dict {variable_name: {metric: value}}
          - "records"     : list of dicts ready to be converted to pd.DataFrame
          - "global"      : dict of global averages (with correction for the smape typo)
        """
        if file:
            self._write_and_print("=== ÉVALUATION DES VARIABLES CONTINUES ===\n", file)
            header = f"{'Variable':<10} | {'MSE':<9} | {'MAE':<9} | {'MAPE (%)':<9} | {'SMAPE (%)':<10} | {'R²':<7} | {'JS':<7} | {'KL':<7} | {'MMD':<7}"
            separator = "-" * len(header)
            self._write_and_print(separator, file)
            self._write_and_print(header, file)
            self._write_and_print(separator, file)

        epsilon = 1e-8
        by_variable = {}
        records = []
        
        mse_cum, mae_cum, mape_cum, smape_cum = 0.0, 0.0, 0.0, 0.0
        r2_cum, js_dist_cum, kl_div_cum, mmd_cum = 0.0, 0.0, 0.0, 0.0

        for idx in self.continuous_indices:
            col_name = self.col_names[idx]
            real_col = self._get_column_vector(self.real_data, idx)
            gen_col = self._get_column_vector(self.gen_data, idx)

            # Point-to-point metrics
            mse = float(mean_squared_error(real_col, gen_col))
            mae = float(mean_absolute_error(real_col, gen_col))
            mape = float(np.mean(np.abs((real_col - gen_col) / (np.abs(real_col) + epsilon))) * 100)
            smape = float(np.mean(2 * np.abs(gen_col - real_col) / (np.abs(real_col) + np.abs(gen_col) + epsilon)) * 100)
            r2 = float(r2_score(real_col, gen_col))

            # Histograms and additive smoothing (to avoid p=0 or grid=0 leading to inf on KL)
            bins = np.histogram_bin_edges(np.concatenate([real_col, gen_col]), bins=50)
            p, _ = np.histogram(real_col, bins=bins, density=False)
            grid, _ = np.histogram(gen_col, bins=bins, density=False)

            p = (p + epsilon) / (np.sum(p) + epsilon * len(p))
            grid = (grid + epsilon) / (np.sum(grid) + epsilon * len(grid))

            js_dist = float(jensenshannon(p, grid))
            kl_div = float(entropy(p, grid))
            mmd = float(self._calculate_mmd(real_col, gen_col))

            # Store metrics for the variable
            metrics_dict = {
                "MSE": mse,
                "MAE": mae,
                "MAPE": mape,
                "SMAPE": smape,
                "R2": r2,
                "JS": js_dist,
                "KL": kl_div,
                "MMD": mmd
            }
            by_variable[col_name] = metrics_dict
            records.append({"variable": col_name, **metrics_dict})

            if file:
                row = f"{col_name:<10} | {mse:<9.4f} | {mae:<9.4f} | {mape:<9.2f} | {smape:<10.2f} | {r2:<7.3f} | {js_dist:<7.4f} | {kl_div:<7.4f} | {mmd:<7.4f}"
                self._write_and_print(row, file)

            mse_cum += mse
            mae_cum += mae
            mape_cum += mape
            smape_cum += smape
            r2_cum += r2
            js_dist_cum += js_dist
            kl_div_cum += kl_div
            mmd_cum += mmd

        if file:
            self._write_and_print(separator, file)

        n_vars = len(self.continuous_indices)
        global_summary = {
            "mse_glob": mse_cum / n_vars,
            "mae_glob": mae_cum / n_vars,
            "mape_glob": mape_cum / n_vars,
            "smape_glob": smape_cum / n_vars,
            "r2_glob": r2_cum / n_vars,
            "js_glob": js_dist_cum / n_vars,
            "kl_glob": kl_div_cum / n_vars,
            "mmd_glob": mmd_cum / n_vars
        }

        return {
            "by_variable": by_variable,
            "records": records,
            "global": global_summary
        }

    def evaluate_categorical_event(self, file=None):
        """
        Calculate metrics for the categorical variable (event_code)
        and return a structured dictionary of results.
        """
        if file:
            self._write_and_print("\n=== CATEGORICAL VARIABLE EVALUATION (event_code) ===\n", file)
        
        real_events = self._get_column_vector(self.real_data, self.event_idx).astype(int)
        gen_events = self._get_column_vector(self.gen_data, self.event_idx).astype(int)
        classes = np.unique(np.concatenate([real_events, gen_events]))
        
        unique_elements, counts_elements = np.unique(real_events, return_counts=True)
        major_class = int(unique_elements[np.argmax(counts_elements)])
        major_count = int(np.max(counts_elements))
        classes_eval = unique_elements[counts_elements > 0]
        
        accuracy_globale = float(np.mean(real_events == gen_events))
        minority_mask = (real_events != major_class)
        
        if np.sum(minority_mask) > 0:
            accuracy_minoritaire = float(np.mean(real_events[minority_mask] == gen_events[minority_mask]))
        else:
            accuracy_minoritaire = float('nan')

        macro_f1_arith = float(f1_score(real_events, gen_events, average='macro'))
        weighted_f1 = float(f1_score(real_events, gen_events, average='weighted'))

        fsg_macro = float(self._calculate_fsg(real_events, gen_events, classes_eval, weights=None))
        support_weights = counts_elements[counts_elements > 0]
        fsg_weighted = float(self._calculate_fsg(real_events, gen_events, classes_eval, weights=support_weights))

        gmean_strict = float(geometric_mean_score(real_events, gen_events, labels=classes_eval, average='multiclass', correction=0))
        gmean_smoothed = float(geometric_mean_score(real_events, gen_events, labels=classes_eval, average='multiclass', correction=1e-3))
        macro_auc_roc, weighted_auc_roc = self._calculate_discrete_multiclass_auc_roc(real_events, gen_events, classes_eval)
        auc_pr_macro = float(self._calculate_multiclass_auc_pr(real_events, gen_events, classes))

        if file:
            self._write_and_print(f"Identified majority class                    : {major_class} (Present {major_count}/{len(real_events)})", file)
            self._write_and_print(f"Global accuracy                              : {accuracy_globale:.4f}", file)
            self._write_and_print(f"Accuracy (excluding majority class)          : {accuracy_minoritaire:.4f}", file)
            self._write_and_print(f"F1-Score (Arithmetic, Macro)                 : {macro_f1_arith:.4f}", file)
            self._write_and_print(f"F1-Score (Arithmetic, Weighted)              : {weighted_f1:.4f}", file)
            self._write_and_print(f"F1-Score (Geometric, Macro)                  : {fsg_macro:.4f}", file)
            self._write_and_print(f"F1-Score (Geometric, Weighted)               : {fsg_weighted:.4f}", file)
            self._write_and_print(f"G-Mean strict                                : {gmean_strict:.4f}", file)
            self._write_and_print(f"G-Mean smoothed (1e-3)                       : {gmean_smoothed:.4f}", file)
            self._write_and_print(f"AUC-ROC (Macro, One-vs-Rest)                 : {macro_auc_roc:.4f}", file)
            self._write_and_print(f"AUC-ROC (Weighted)                           : {weighted_auc_roc:.4f}", file)
            self._write_and_print(f"AUC-PR  (Macro, One-vs-Rest)                 : {auc_pr_macro:.4f}", file)
            
            self._write_and_print("\nDetailed class report :", file)
            report = classification_report(real_events, gen_events, zero_division=0)
            self._write_and_print(report, file)

        return {
            "accuracy_globale": accuracy_globale,
            "accuracy_minoritaire": accuracy_minoritaire,
            "macro_f1_arith": macro_f1_arith,
            "weighted_f1": weighted_f1,
            "fsg_macro": fsg_macro,
            "fsg_weighted": fsg_weighted,
            "gmean_strict": gmean_strict,
            "gmean_smoothed": gmean_smoothed,
            "macro_auc_roc": float(macro_auc_roc),
            "weighted_auc_roc": float(weighted_auc_roc),
            "auc_pr_macro": auc_pr_macro
        }

    def distribution_plots(self, file_img="population_distribution.png", file_txt="evaluation.txt", display_screen=True):
        """
        Generates distribution histograms for each continuous variable,
        calculates the overlap percentage and writes it to the final report.
        """
        cols_grid = int(np.ceil(len(self.col_names) / 2))
        fig, axes = plt.subplots(2, cols_grid, figsize=(16, 9))
        fig.suptitle(f"Comparison of Distributions (Population : {self.num_patients} patients)", fontsize=16)
        axes = axes.flatten()
        
        full_txt_path = f"{self.path}{file_txt}"
        
        with open(full_txt_path, "a", encoding="utf-8") as f:
            f.write("\n=== OVERLAP OF DISTRIBUTIONS (INTERSECTION OF DENSITIES) ===\n\n")

        for plot_idx, idx in enumerate(self.col_indices):
            col_name = self.col_names[idx]
            real_flat = self._get_column_vector(self.real_data, idx)
            gen_flat = self._get_column_vector(self.gen_data, idx)
            
            min_val = min(real_flat.min(), gen_flat.min())
            max_val = max(real_flat.max(), gen_flat.max())
            bins = np.linspace(min_val, max_val, 50)
            
            hist_real, _ = np.histogram(real_flat, bins=bins)
            hist_gen, _ = np.histogram(gen_flat, bins=bins)
            
            prob_real = hist_real / len(real_flat)
            prob_gen = hist_gen / len(gen_flat)
            
            overlap = np.sum(np.minimum(prob_real, prob_gen))
            overlap_pct = overlap * 100

            weights_real = np.ones_like(real_flat) / len(real_flat)
            weights_gen = np.ones_like(gen_flat) / len(gen_flat)
            
            axes[plot_idx].hist(real_flat, bins=bins, alpha=0.5, weights=weights_real, color='blue', label='Réel')
            axes[plot_idx].hist(gen_flat, bins=bins, alpha=0.5, weights=weights_gen, color='red', label='Généré')
            
            axes[plot_idx].set_title(f"{col_name} (Overlap : {overlap_pct:.1f}%)")
            axes[plot_idx].legend()

            with open(full_txt_path, "a", encoding="utf-8") as f:
                f.write(f"   - {col_name:<10} : {overlap_pct:.1f}%\n")
        
        for i in range(len(self.col_names), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        os.makedirs(self.path, exist_ok=True)
        plt.savefig(f"{self.path}{file_img}")
        print(f"[INFO] Distribution plot saved at : {self.path}{file_img}")
        
        if display_screen and not os.environ.get("NO_PLOT"):
            plt.show()

    def run_full_analysis(self, plot=False):
        """
        Executes the entire protocol and generates the text file and image.
        """
        os.makedirs(self.path, exist_ok=True)
        full_txt_path = f"{self.path}{self.output_file}"
        
        with open(full_txt_path, "w", encoding="utf-8") as file:
            self._write_and_print("====================================================================================", file)
            self._write_and_print("                               EVALUATION REPORT                                    ", file)
            self._write_and_print("====================================================================================\n", file)
            
            cont_metrics = self.evaluate_continuous_variables(file)
            
            cat_metrics = self.evaluate_categorical_event(file)

        if plot:
            self.distribution_plots()

        print(f"[INFO] Global analysis completed. Results in '{self.path}'.")

        return cont_metrics, cat_metrics