class MetricsReporter:
    @staticmethod
    def show_confusion_matrix(interp, labels):
        cm = interp.confusion_matrix()
        print("\n🔍 Confusion Matrix:")
        print(f"{'':>12} {' '.join([f'{label:>8}' for label in labels])}")
        for i, row in enumerate(cm):
            print(f"{labels[i]:>12} {' '.join([f'{n:>8}' for n in row])}")

        correct = sum(cm[i][i] for i in range(len(labels)))
        total = sum(sum(row) for row in cm)
        accuracy = correct / total if total > 0 else 0

        print(f"\n✅ Total samples: {total}")
        print(f"✅ Correct predictions: {correct}")
        print(f"❌ Misclassifications: {total - correct}")
        print(f"📊 Overall Accuracy: {accuracy:.2%}")
