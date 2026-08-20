import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, f1_score, 
    classification_report, r2_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

class DatasetEvaluator:
    def __init__(self, real_data, gen_data, col_names, path_dir="checkpoints/"):
        """
        Initialise l'évaluateur.
        :param real_data: np.array contenant les données réelles (2D ou 3D).
        :param gen_data: np.array contenant les données générées (2D ou 3D).
        :param col_names: liste des noms de colonnes dans l'ordre.
        :param path_dir: dossier de sauvegarde.
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
        Écrit dans le fichier et dans la console.
        """
        print(text)
        file.write(text + "\n")

    def _get_column_vector(self, data, idx):
        """
        Extrait une variable sous forme de vecteur 1D (2D ou 3D).
        """
        if self.is_3d:
            return data[:, :, idx].flatten()
        else:
            return data[:, idx].flatten()

    def _calculate_mmd(self, x, y, gamma=1.0):
        """
        Calcule une approximation de la Maximum Mean Discrepancy (MMD).
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
        Calcule l'AUC-PR macro en encodant One-vs-Rest et en ignorant les classes vides.
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

    def evaluate_continuous_variables(self, file):
        """
        Calcule l'ensemble des métriques de régression sous forme de grand tableau.
        """
        self._write_and_print("=== ÉVALUATION DES VARIABLES CONTINUES ===\n", file)
        
        header = f"{'Variable':<10} | {'MSE':<9} | {'MAE':<9} | {'MAPE (%)':<9} | {'SMAPE (%)':<10} | {'R²':<7} | {'JS':<7} | {'KL':<7} | {'MMD':<7}"
        separator = "-" * len(header)
        
        self._write_and_print(separator, file)
        self._write_and_print(header, file)
        self._write_and_print(separator, file)
        epsilon = 1e-5
        
        for idx in self.continuous_indices:
            col_name = self.col_names[idx]
            real_col = self._get_column_vector(self.real_data, idx)
            gen_col = self._get_column_vector(self.gen_data, idx)
            
            mse = mean_squared_error(real_col, gen_col)
            mae = mean_absolute_error(real_col, gen_col)
            mape = np.mean(np.abs((real_col - gen_col) / (np.abs(real_col) + epsilon))) * 100
            smape = np.mean(2 * np.abs(gen_col - real_col) / (np.abs(real_col) + np.abs(gen_col) + epsilon)) * 100
            r2 = r2_score(real_col, gen_col)
            
            bins = np.histogram_bin_edges(np.concatenate([real_col, gen_col]), bins=50)
            p, _ = np.histogram(real_col, bins=bins, density=True)
            grid, _ = np.histogram(gen_col, bins=bins, density=True)
            
            p = p / (np.sum(p) + epsilon)
            grid = grid / (np.sum(grid) + epsilon)
            
            js_dist = jensenshannon(p, grid)
            kl_div = entropy(p, grid)
            mmd = self._calculate_mmd(real_col, gen_col)

            row = f"{col_name:<10} | {mse:<9.4f} | {mae:<9.4f} | {mape:<9.2f} | {smape:<10.2f} | {r2:<7.3f} | {js_dist:<7.4f} | {kl_div:<7.4f} | {mmd:<7.4f}"
            self._write_and_print(row, file)
            
        self._write_and_print(separator, file)

    def evaluate_categorical_event(self, file):
        """
        Calcule les métriques avancées pour la variable catégorielle (event_code).
        """
        self._write_and_print("\n=== ÉVALUATION DE LA VARIABLE CATÉGORIELLE (event_code) ===\n", file)
        
        real_events = self._get_column_vector(self.real_data, self.event_idx).astype(int)
        gen_events = self._get_column_vector(self.gen_data, self.event_idx).astype(int)
        classes = np.unique(np.concatenate([real_events, gen_events]))
        
        unique_elements, counts_elements = np.unique(real_events, return_counts=True)
        major_class = unique_elements[np.argmax(counts_elements)]
        major_count = np.max(counts_elements)
        
        accuracy_globale = np.mean(real_events == gen_events)
        minority_mask = (real_events != major_class)
        
        if np.sum(minority_mask) > 0:
            accuracy_minoritaire = np.mean(real_events[minority_mask] == gen_events[minority_mask])
        else:
            accuracy_minoritaire = float('nan')

        macro_f1_arith = f1_score(real_events, gen_events, average='macro')
        weighted_f1 = f1_score(real_events, gen_events, average='weighted')
        
        f1_per_class = f1_score(real_events, gen_events, average=None, zero_division=0)
        class_to_support = dict(zip(unique_elements, counts_elements))
        valid_f1_scores = [f1_per_class[np.where(classes == cls)[0][0]] for cls in unique_elements if class_to_support.get(cls, 0) > 0]
        macro_f1_filtre = np.mean(valid_f1_scores) if len(valid_f1_scores) > 0 else 0.0
        macro_f1_geom = np.exp(np.mean(np.log(f1_per_class + 1e-10))) if len(f1_per_class) > 0 else 0.0
        auc_pr_macro = self._calculate_multiclass_auc_pr(real_events, gen_events, classes)

        self._write_and_print(f"Classe majoritaire identifiée                : {major_class} (Présente {major_count}/{len(real_events)})", file)
        self._write_and_print(f"Accuracy Globale                             : {accuracy_globale:.4f}", file)
        self._write_and_print(f"Accuracy Hors Classe Majoritaire             : {accuracy_minoritaire:.4f}", file)
        self._write_and_print(f"F1-Score Arithmétique (Macro Global)         : {macro_f1_arith:.4f}", file)
        self._write_and_print(f"F1-Score Macro Filtré (Support > 0)          : {macro_f1_filtre:.4f}", file)
        self._write_and_print(f"F1-Score Arithmétique (Pondéré / Weighted)   : {weighted_f1:.4f}", file)
        self._write_and_print(f"F1-Score Géométrique (G-Mean des classes)    : {macro_f1_geom:.4f}", file)
        self._write_and_print(f"AUC-PR  (Macro, One-vs-Rest)                 : {auc_pr_macro:.4f}", file)
        
        self._write_and_print("\nRapport détaillé par classe :", file)
        report = classification_report(real_events, gen_events, zero_division=0)
        self._write_and_print(report, file)

    def distribution_plots(self, file_img="population_distribution.png", file_txt="evaluation.txt", display_screen=True):
        """
        Génère les histogrammes de distribution pour chaque variable continue,
        calcule le pourcentage d'overlap et l'écrit dans le rapport final.
        """
        cols_grid = int(np.ceil(len(self.col_names) / 2))
        fig, axes = plt.subplots(2, cols_grid, figsize=(16, 9))
        fig.suptitle(f"Comparaison des Distributions (Population : {self.num_patients} patients)", fontsize=16)
        axes = axes.flatten()
        
        full_txt_path = f"{self.path}{file_txt}"
        
        with open(full_txt_path, "a", encoding="utf-8") as f:
            f.write("\n=== OVERLAP DES DISTRIBUTIONS (INTERSECTION DES DENSITÉS) ===\n\n")

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
        print(f"[INFO] Graphique de distribution sauvegardé dans : {self.path}{file_img}")
        
        if display_screen and not os.environ.get("NO_PLOT"):
            plt.show()

    def run_full_analysis(self):
        """
        Exécute l'ensemble du protocole et génère le fichier texte et l'image.
        """
        os.makedirs(self.path, exist_ok=True)
        full_txt_path = f"{self.path}{self.output_file}"
        
        with open(full_txt_path, "w", encoding="utf-8") as file:
            self._write_and_print("====================================================================================", file)
            self._write_and_print("                         RAPPORT D'ÉVALUATION AVANCÉ                                ", file)
            self._write_and_print("====================================================================================\n", file)
            
            self.evaluate_continuous_variables(file)
            
            self.evaluate_categorical_event(file)
            
        self.distribution_plots()
        
        print(f"[INFO] Analyse globale terminée. Résultats dans '{self.path}'.")