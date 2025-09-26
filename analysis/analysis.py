import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import argparse
import sys
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MedicalSegmentationAnalyzer:
    def __init__(self):
        self.data = {}
        self.structures = None
        self.scenarios = None

    def load_data(self, train_file, val_file, test_file):
        """Load the three evaluation files"""
        with open(train_file, 'r') as f:
            self.data['train'] = json.load(f)
        with open(val_file, 'r') as f:
            self.data['validation'] = json.load(f)
        with open(test_file, 'r') as f:
            self.data['test'] = json.load(f)

        # Get structure names and scenarios
        self.structures = list(self.data['train']['presence_per_structure'].keys())
        self.scenarios = list(self.data['train']['scenario_based'].keys())

    def create_presence_accuracy_comparison(self):
        """Compare presence detection accuracy across structures and datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Prepare data for plotting
        accuracy_data = []
        for dataset in ['train', 'validation', 'test']:
            for structure in self.structures:
                accuracy = self.data[dataset]['presence_per_structure'][structure]['accuracy']
                accuracy_data.append({
                    'Dataset': dataset.capitalize(),
                    'Structure': structure,
                    'Accuracy': accuracy
                })

        df_accuracy = pd.DataFrame(accuracy_data)

        # Bar plot comparison
        sns.barplot(data=df_accuracy, x='Structure', y='Accuracy', hue='Dataset', ax=ax1)
        ax1.set_title('Presence Detection Accuracy by Structure', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Anatomical Structure', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])
        ax1.set_ylim([0, 1.02])

        # Overall accuracy comparison
        overall_acc = []
        for dataset in ['train', 'validation', 'test']:
            overall_acc.append({
                'Dataset': dataset.capitalize(),
                'Overall_Accuracy': self.data[dataset]['presence_overall_accuracy']
            })

        df_overall = pd.DataFrame(overall_acc)
        bars = ax2.bar(df_overall['Dataset'], df_overall['Overall_Accuracy'])
        ax2.set_title('Overall Presence Detection Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])

        # Add value labels on bars
        for bar, acc in zip(bars, df_overall['Overall_Accuracy']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig

    def create_segmentation_dice_comparison(self):
        """Compare Dice scores across structures and datasets"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Prepare data for Dice scores
        dice_data = []
        for dataset in ['train', 'validation', 'test']:
            for structure in self.structures:
                dice = self.data[dataset]['segmentation_per_structure'][structure]['mean_dice']
                std_dice = self.data[dataset]['segmentation_per_structure'][structure]['std_dice']
                dice_data.append({
                    'Dataset': dataset.capitalize(),
                    'Structure': structure,
                    'Mean_Dice': dice,
                    'Std_Dice': std_dice
                })

        df_dice = pd.DataFrame(dice_data)

        # Box plot style comparison with error bars
        pivot_dice = df_dice.pivot(index='Structure', columns='Dataset', values='Mean_Dice')
        pivot_std = df_dice.pivot(index='Structure', columns='Dataset', values='Std_Dice')

        x_pos = np.arange(len(self.structures))
        width = 0.25

        for i, dataset in enumerate(['Train', 'Validation', 'Test']):
            means = pivot_dice[dataset].values
            stds = pivot_std[dataset].values
            ax1.bar(x_pos + i * width, means, width, yerr=stds,
                    label=dataset, capsize=3, alpha=0.8)

        ax1.set_xlabel('Anatomical Structure', fontsize=12)
        ax1.set_ylabel('Dice Score', fontsize=12)
        ax1.set_title('Segmentation Dice Scores by Structure', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels(self.structures, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Overall segmentation performance
        overall_dice = []
        overall_iou = []
        for dataset in ['train', 'validation', 'test']:
            overall_dice.append({
                'Dataset': dataset.capitalize(),
                'Metric': 'Dice',
                'Score': self.data[dataset]['segmentation_overall']['mean_dice']
            })
            overall_iou.append({
                'Dataset': dataset.capitalize(),
                'Metric': 'IoU',
                'Score': self.data[dataset]['segmentation_overall']['mean_iou']
            })

        df_overall_seg = pd.DataFrame(overall_dice + overall_iou)
        sns.barplot(data=df_overall_seg, x='Dataset', y='Score', hue='Metric', ax=ax2)
        ax2.set_title('Overall Segmentation Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_scenario_based_analysis(self):
        """Analyze performance across different clinical scenarios"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # Prepare scenario data
        scenario_data = []
        for dataset in ['train', 'validation', 'test']:
            for scenario in self.scenarios:
                if scenario in self.data[dataset]['scenario_based']:
                    scenario_info = self.data[dataset]['scenario_based'][scenario]
                    scenario_data.append({
                        'Dataset': dataset.capitalize(),
                        'Scenario': scenario,
                        'Accuracy': scenario_info['accuracy'],
                        'Num_Samples': scenario_info['num_samples'],
                        'Avg_Structures_Present': scenario_info['avg_structures_present'],
                        'Avg_Structures_Predicted': scenario_info['avg_structures_predicted']
                    })

        df_scenarios = pd.DataFrame(scenario_data)

        # Scenario accuracy comparison
        pivot_scenario_acc = df_scenarios.pivot(index='Scenario', columns='Dataset', values='Accuracy')
        pivot_scenario_acc.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Scenario-Based Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Dataset')
        ax1.grid(True, alpha=0.3)

        # Sample distribution across scenarios
        pivot_samples = df_scenarios.pivot(index='Scenario', columns='Dataset', values='Num_Samples')
        pivot_samples.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Sample Distribution Across Scenarios', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Dataset')
        ax2.grid(True, alpha=0.3)

        # Structure prediction accuracy
        df_scenarios['Prediction_Error'] = abs(df_scenarios['Avg_Structures_Present'] -
                                               df_scenarios['Avg_Structures_Predicted'])

        sns.boxplot(data=df_scenarios, x='Dataset', y='Prediction_Error', ax=ax3)
        ax3.set_title('Structure Count Prediction Error by Dataset', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Average Prediction Error', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, None])  # Start from 0, auto-scale upper limit

        # Challenging scenarios (low accuracy)
        challenging_scenarios = df_scenarios[df_scenarios['Accuracy'] < 0.8]
        if not challenging_scenarios.empty:
            sns.scatterplot(data=challenging_scenarios, x='Num_Samples', y='Accuracy',
                            hue='Dataset', style='Scenario', s=100, ax=ax4)
            ax4.set_title('Challenging Scenarios (Accuracy < 80%)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Number of Samples', fontsize=12)
            ax4.set_ylabel('Accuracy', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1.0])
        else:
            ax4.text(0.5, 0.5, 'No challenging scenarios\n(all accuracies ≥ 80%)',
                     transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Challenging Scenarios Analysis', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def create_performance_heatmap(self):
        """Create heatmaps showing performance metrics across structures and datasets"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # Prepare data for heatmaps
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        for i, metric in enumerate(metrics):
            ax = [ax1, ax2, ax3, ax4][i]

            # Create matrix for heatmap
            matrix_data = []
            for structure in self.structures:
                row = []
                for dataset in ['train', 'validation', 'test']:
                    value = self.data[dataset]['presence_per_structure'][structure][metric]
                    row.append(value)
                matrix_data.append(row)

            # Create heatmap with colormap starting from 0
            sns.heatmap(matrix_data,
                        xticklabels=['Train', 'Validation', 'Test'],
                        yticklabels=self.structures,
                        annot=True, fmt='.3f', cmap='RdYlBu_r',
                        vmin=0, vmax=1,  # Set colormap range from 0 to 1
                        ax=ax, cbar_kws={'shrink': .8})
            ax.set_title(f'{metric.replace("_", " ").title()} Heatmap',
                         fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def generate_summary_report(self):
        """Generate a text summary of key findings"""
        print("=" * 70)
        print("MEDICAL SEGMENTATION MODEL EVALUATION SUMMARY")
        print("=" * 70)

        # Overall performance
        print("\n1. OVERALL PERFORMANCE:")
        print("-" * 25)
        for dataset in ['train', 'validation', 'test']:
            overall_acc = self.data[dataset]['presence_overall_accuracy']
            overall_dice = self.data[dataset]['segmentation_overall']['mean_dice']
            overall_iou = self.data[dataset]['segmentation_overall']['mean_iou']
            print(f"{dataset.upper():>10}: Presence Acc: {overall_acc:.4f}, "
                  f"Dice: {overall_dice:.4f}, IoU: {overall_iou:.4f}")

        # Best and worst performing structures
        print("\n2. STRUCTURE PERFORMANCE (Test Set):")
        print("-" * 35)

        test_dice_scores = []
        for structure in self.structures:
            dice = self.data['test']['segmentation_per_structure'][structure]['mean_dice']
            test_dice_scores.append((structure, dice))

        test_dice_scores.sort(key=lambda x: x[1], reverse=True)

        print("Best performing structures (Dice score):")
        for structure, dice in test_dice_scores[:3]:
            print(f"  {structure:>15}: {dice:.4f}")

        print("\nWorst performing structures (Dice score):")
        for structure, dice in test_dice_scores[-3:]:
            print(f"  {structure:>15}: {dice:.4f}")

        # Challenging scenarios
        print("\n3. CHALLENGING SCENARIOS (Test Set):")
        print("-" * 35)

        test_scenarios = []
        for scenario in self.scenarios:
            if scenario in self.data['test']['scenario_based']:
                acc = self.data['test']['scenario_based'][scenario]['accuracy']
                samples = self.data['test']['scenario_based'][scenario]['num_samples']
                test_scenarios.append((scenario, acc, samples))

        test_scenarios.sort(key=lambda x: x[1])

        print("Most challenging scenarios:")
        for scenario, acc, samples in test_scenarios[:5]:
            print(f"  {scenario:>25}: {acc:.3f} (n={samples})")

        print("\n4. RECOMMENDATIONS:")
        print("-" * 20)

        # Find structures with high variance
        high_var_structures = []
        for structure in self.structures:
            test_std = self.data['test']['segmentation_per_structure'][structure]['std_dice']
            if test_std > 0.15:
                high_var_structures.append((structure, test_std))

        if high_var_structures:
            print("• Consider improving consistency for high-variance structures:")
            for structure, std in sorted(high_var_structures, key=lambda x: x[1], reverse=True)[:3]:
                print(f"  - {structure} (std: {std:.3f})")

        # Check for overfitting
        train_dice = self.data['train']['segmentation_overall']['mean_dice']
        test_dice = self.data['test']['segmentation_overall']['mean_dice']
        dice_gap = train_dice - test_dice

        if dice_gap > 0.05:
            print(f"• Potential overfitting detected (Train-Test Dice gap: {dice_gap:.4f})")
        else:
            print(f"• Good generalization (Train-Test Dice gap: {dice_gap:.4f})")

        print("=" * 70)


def main():
    import argparse
    import sys
    import os

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Analyze medical segmentation model performance across train/validation/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python segmentation_analysis.py train.json val.json test.json
  python segmentation_analysis.py --train train_metrics.json --val val_metrics.json --test test_metrics.json
  python segmentation_analysis.py train.json val.json test.json --output-dir ./plots --no-show
        """)

    # Positional arguments (backward compatibility)
    parser.add_argument('files', nargs='*',
                        help='JSON files in order: train_file val_file test_file')

    # Named arguments (more explicit)
    parser.add_argument('--train', type=str,
                        help='Training set metrics JSON file')
    parser.add_argument('--val', '--validation', type=str,
                        help='Validation set metrics JSON file')
    parser.add_argument('--test', type=str,
                        help='Test set metrics JSON file')

    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='./plots',
                        help='Directory to save plots (default: ./plots)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots interactively')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots to files')
    parser.add_argument('--no-report', action='store_true',
                        help='Do not generate text summary report')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='Output format for saved plots (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plots (default: 300)')

    args = parser.parse_args()

    # Determine file paths
    train_file = val_file = test_file = None

    if args.files and len(args.files) == 3:
        train_file, val_file, test_file = args.files
    elif args.train and args.val and args.test:
        train_file, val_file, test_file = args.train, args.val, args.test
    else:
        print("Error: Must provide exactly 3 JSON files")
        print("\nOption 1 - Positional arguments:")
        print("  python segmentation_analysis.py train.json val.json test.json")
        print("\nOption 2 - Named arguments:")
        print("  python segmentation_analysis.py --train train.json --val val.json --test test.json")
        sys.exit(1)

    # Check if files exist
    for file_path, file_type in [(train_file, 'train'), (val_file, 'validation'), (test_file, 'test')]:
        if not os.path.exists(file_path):
            print(f"Error: {file_type} file '{file_path}' not found")
            sys.exit(1)

    # Create output directory if saving plots
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    print("Medical Segmentation Analysis")
    print("=" * 50)
    print(f"Train file:      {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Test file:       {test_file}")
    print()

    try:
        # Initialize analyzer and load data
        print("Loading data...")
        analyzer = MedicalSegmentationAnalyzer()
        analyzer.load_data(train_file, val_file, test_file)
        print("✓ Data loaded successfully")

        # Generate plots
        print("\nGenerating visualizations...")

        print("  • Creating presence accuracy comparison...")
        fig1 = analyzer.create_presence_accuracy_comparison()

        print("  • Creating segmentation Dice comparison...")
        fig2 = analyzer.create_segmentation_dice_comparison()

        print("  • Creating scenario-based analysis...")
        fig3 = analyzer.create_scenario_based_analysis()

        print("  • Creating performance heatmaps...")
        fig4 = analyzer.create_performance_heatmap()

        # Save plots
        if not args.no_save:
            print(f"\nSaving plots to {args.output_dir}/...")
            plot_files = [
                (fig1, 'presence_accuracy_comparison'),
                (fig2, 'segmentation_dice_comparison'),
                (fig3, 'scenario_based_analysis'),
                (fig4, 'performance_heatmap')
            ]

            for fig, name in plot_files:
                filename = f"{name}.{args.format}"
                filepath = os.path.join(args.output_dir, filename)
                fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"  ✓ {filename}")

        # Show plots interactively
        if not args.no_show:
            print("\nDisplaying plots...")
            plt.show()

        # Generate summary report
        if not args.no_report:
            print("\nGenerating summary report...")
            analyzer.generate_summary_report()

            # Optionally save report to file
            if not args.no_save:
                report_file = os.path.join(args.output_dir, 'analysis_report.txt')
                import io
                import contextlib

                # Capture the report output
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    analyzer.generate_summary_report()
                report_content = f.getvalue()

                # Save to file
                with open(report_file, 'w') as rf:
                    rf.write(report_content)
                print(f"✓ Analysis report saved to: {report_file}")

        print("\nAnalysis completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected field in JSON - {e}")
        print("Please ensure your JSON files have the correct structure.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
