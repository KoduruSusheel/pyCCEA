import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models, datasets
import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Tuple, List, Optional

# --- Configuration (MAXIMUM PERFORMANCE SETTINGS) ---
STAGE1_EPOCHS = 12  # Training only the classifier (Frozen layers)
STAGE2_EPOCHS = 8   # Fine-tuning the whole model (Unfrozen layers)
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS # Max 20 epochs
BATCH_SIZE = 32
LR_STAGE1 = 0.001
LR_STAGE2 = 0.0001
IMAGE_SIZE = 224
NUM_FEATURES = 2048 # Feature vector size for ResNet-101/50 before the FC layer
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. PyCCEA Implementation (Feature Selection)
# =============================================================================
class PyCCEA:
    """A simplified implementation of cooperative co-evolutionary algorithms for feature selection."""
    def __init__(self, population_size=50, generations=80, tournament_size=3,
                 mutation_rate=0.1, crossover_rate=0.8, feature_ratio=0.6):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.feature_ratio = feature_ratio

    def initialize_population(self, num_features):
        population = []
        subset_size = max(1, int(num_features * self.feature_ratio))
        for _ in range(self.population_size):
            individual = np.zeros(num_features, dtype=bool)
            selected_indices = np.random.choice(num_features, subset_size, replace=False)
            individual[selected_indices] = True
            population.append(individual)
        return population

    def tournament_selection(self, population, fitness):
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index])
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1)-1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = not mutated[i]
        return mutated

    def cooperative_evaluation(self, population, feature_importance_scores):
        fitness_scores = []
        for individual in population:
            if np.sum(individual) == 0:
                fitness = 0.0
            else:
                fitness = np.sum(feature_importance_scores[individual])
                feature_count = np.sum(individual)
                penalty = abs(feature_count - len(individual) * self.feature_ratio) / len(individual)
                fitness *= (1 - penalty * 0.1)
            fitness_scores.append(fitness)
        return fitness_scores

    def evolve(self, feature_importance_scores):
        num_features = len(feature_importance_scores)
        population = self.initialize_population(num_features)

        best_individual = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            fitness_scores = self.cooperative_evaluation(population, feature_importance_scores)

            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()

            selected = self.tournament_selection(population, fitness_scores)

            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                else:
                    new_population.append(self.mutate(selected[i]))

            population = new_population

            if generation % 40 == 0 and generation > 0:
                print(f"PyCCEA Generation {generation}: Best Fitness = {best_fitness:.4f}")

        return best_individual

class PyCCEAFeatureSelector:
    def __init__(self):
        self.ccea = PyCCEA(population_size=50, generations=80, feature_ratio=0.6)
        self.selected_features = None
        self.num_selected_features = 0

    def compute_feature_importance(self, dataloader, device, feature_extractor, num_features=NUM_FEATURES):
        """Compute feature importance using the ResNet feature map variance."""
        print("Computing feature importance using ResNet feature map variance...")

        feature_extractor.eval()
        all_features = []
        sample_batches = 10

        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= sample_batches:
                    break
                inputs = inputs.to(device)

                features = feature_extractor(inputs)
                features = features.view(features.size(0), -1)
                all_features.append(features.cpu().numpy())

        if all_features:
            all_features = np.concatenate(all_features, axis=0)
            importance_scores = np.var(all_features, axis=0)

            if np.max(importance_scores) > 0:
                importance_scores = importance_scores / np.max(importance_scores)

            importance_scores = importance_scores[:num_features]

        else:
            print("Warning: Could not extract features. Using uniform importance.")
            importance_scores = np.ones(num_features)

        return importance_scores

    def select_features(self, dataloader, device, feature_extractor, num_features=NUM_FEATURES):
        """Select features using PyCCEA"""
        print("\n" + "="*50)
        print("STARTING PYCCEA FEATURE SELECTION")
        print("="*50)

        importance_scores = self.compute_feature_importance(dataloader, device, feature_extractor, num_features)

        selected_mask = self.ccea.evolve(importance_scores)
        self.selected_features = selected_mask
        self.num_selected_features = int(np.sum(selected_mask))

        print(f"✅ PyCCEA selected {self.num_selected_features}/{num_features} features.")

        return selected_mask, self.num_selected_features

class FeatureSelectedResNet(nn.Module):
    def __init__(self, base_model, num_classes, feature_selector):
        super(FeatureSelectedResNet, self).__init__()
        # Feature extractor is ResNet up to the last AvgPool
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_selector = feature_selector
        self.original_num_features = base_model.fc.in_features

        self.classifier = nn.Linear(self.original_num_features, num_classes)

    def update_classifier(self, num_selected_features):
        self.classifier = nn.Linear(num_selected_features, self.classifier.out_features)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # Apply feature selection mask
        if self.feature_selector.selected_features is not None:
            selected_mask = torch.tensor(self.feature_selector.selected_features,
                                       dtype=torch.bool, device=features.device)
            available_features = min(features.size(1), len(selected_mask))
            selected_mask = selected_mask[:available_features]
            features = features[:, :available_features]
            features = features[:, selected_mask]

        output = self.classifier(features)
        return output

# =============================================================================
# 2. Dataset and Utility Functions (LOCAL DATASET PATHS)
# =============================================================================

def get_local_data_root(dataset_name: str) -> Optional[str]:
    """Resolve local dataset root under ./data based on expected folder structure.

    Expected after extraction:
    - uc_merced -> data/UCMerced_LandUse/Images
    - lc25000   -> data/lung_colon_image_set/(Train and Validation Set, Test Set)
    - plants    -> data/split_ttv_dataset_type_of_plants/(Train_Set_Folder, Validation_Set_Folder, Test_Set_Folder)
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "data")

    if dataset_name == "uc_merced":
        images_dir = os.path.join(data_root, "UCMerced_LandUse", "Images")
        if os.path.isdir(images_dir):
            return images_dir
        for root, dirs, _ in os.walk(data_root):
            if "Images" in dirs and ("UCMerced" in root or "LandUse" in root):
                return os.path.join(root, "Images")

    elif dataset_name == "aid":
        # AID dataset is expected as data/AID/<class_folders> (already in the workspace)
        aid_dir = os.path.join(data_root, "AID")
        if os.path.isdir(aid_dir):
            return aid_dir
        # fallback: try to discover any folder named AID or with many class subfolders
        for root, dirs, _ in os.walk(data_root):
            if os.path.basename(root).lower() == 'aid' or (len(dirs) >= 10 and 'airport' in (d.lower() for d in dirs)):
                return root

    elif dataset_name == "lc25000":
        candidate = os.path.join(data_root, "lung_colon_image_set")
        if os.path.isdir(os.path.join(candidate, "Train and Validation Set")) and os.path.isdir(os.path.join(candidate, "Test Set")):
            return candidate
        for root, dirs, _ in os.walk(data_root):
            if "Train and Validation Set" in dirs and "Test Set" in dirs and ("lung" in root.lower() or "colon" in root.lower() or "lc25000" in root.lower()):
                return root

    elif dataset_name == "plants":
        candidate = os.path.join(data_root, "split_ttv_dataset_type_of_plants")
        if all(os.path.isdir(os.path.join(candidate, d)) for d in ["Train_Set_Folder", "Validation_Set_Folder", "Test_Set_Folder"]):
            return candidate
        for root, dirs, _ in os.walk(data_root):
            needed = {"Train_Set_Folder", "Validation_Set_Folder", "Test_Set_Folder"}
            if needed.issubset(set(dirs)) and ("plants" in root.lower() or "ttv" in root.lower()):
                return root

    print(f"🚨 Could not locate local data for '{dataset_name}'. Ensure you've extracted zips into the 'data' folder.")
    return None

class TransformDataset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to subsets/splits lazily."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Logic to extract image/label from subset/split
        img_info, label = self.dataset[idx]

        if isinstance(self.dataset, torch.utils.data.Subset):
             # For random_split subsets of ImageFolder
             img_path, _ = self.dataset.dataset.samples[self.dataset.indices[idx]]
             img = Image.open(img_path).convert('RGB')
        elif isinstance(self.dataset, datasets.ImageFolder):
            # For direct ImageFolder loading (e.g., test set)
            img_path, _ = self.dataset.samples[idx]
            img = Image.open(img_path).convert('RGB')
        else:
            # Fallback for unexpected structure
            img = img_info

        if self.transform:
            img = self.transform(img)
        return img, label

def get_data_loaders(data_root, dataset_name):
    """Applies transformations and creates DataLoaders with specific splits."""

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if dataset_name == "lc25000":
        # LC25000: Uses the pre-split directories ('Train and Validation Set', 'Test Set')
        train_val_dir = os.path.join(data_root, "Train and Validation Set")
        test_dir = os.path.join(data_root, "Test Set")

        full_train_val_dataset = datasets.ImageFolder(train_val_dir, transform=None)
        test_dataset_raw = datasets.ImageFolder(test_dir, transform=None)

        # Split Train/Validation (90/10 of the 'Train and Validation Set')
        train_size = int(0.9 * len(full_train_val_dataset))
        val_size = len(full_train_val_dataset) - train_size
        train_dataset_subset, val_dataset_subset = random_split(full_train_val_dataset, [train_size, val_size])

        train_dataset = TransformDataset(train_dataset_subset, transform=data_transforms['train'])
        val_dataset = TransformDataset(val_dataset_subset, transform=data_transforms['val_test'])
        test_dataset = TransformDataset(test_dataset_raw, transform=data_transforms['val_test'])

        class_names = full_train_val_dataset.classes
        num_classes = len(class_names)

    elif dataset_name == "plants":
        # Plants dataset has explicit Train/Val/Test folders
        train_dir = os.path.join(data_root, "Train_Set_Folder")
        val_dir = os.path.join(data_root, "Validation_Set_Folder")
        test_dir = os.path.join(data_root, "Test_Set_Folder")

        train_dataset_raw = datasets.ImageFolder(train_dir, transform=None)
        val_dataset_raw = datasets.ImageFolder(val_dir, transform=None)
        test_dataset_raw = datasets.ImageFolder(test_dir, transform=None)

        class_names = train_dataset_raw.classes
        num_classes = len(class_names)

        train_dataset = TransformDataset(train_dataset_raw, transform=data_transforms['train'])
        val_dataset = TransformDataset(val_dataset_raw, transform=data_transforms['val_test'])
        test_dataset = TransformDataset(test_dataset_raw, transform=data_transforms['val_test'])

    else:
        # AID / UC Merced: Manual 70/10/20 Split
        # The data_root is now correctly pointing to the folder containing the class folders
        full_dataset_raw = datasets.ImageFolder(data_root, transform=None)
        num_classes = len(full_dataset_raw.classes)
        total_size = len(full_dataset_raw)

        train_size = int(0.70 * total_size)
        val_size = int(0.10 * total_size)
        test_size = total_size - train_size - val_size

        # Handle the edge case where the splits might be zero due to missing classes
        if total_size == 0 or train_size == 0 or val_size == 0:
             print("🚨 Error: Dataset is empty or split size is zero. Check data path.")
             return None, None, None, 0, []


        train_subset, val_subset, test_subset = random_split(
            full_dataset_raw, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        train_dataset = TransformDataset(train_subset, transform=data_transforms['train'])
        val_dataset = TransformDataset(val_subset, transform=data_transforms['val_test'])
        test_dataset = TransformDataset(test_subset, transform=data_transforms['val_test'])

        class_names = full_dataset_raw.classes

    print(f"📝 Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}. Classes: {num_classes}")

    # Use num_workers=0 on Windows to avoid multiprocessing issues; change if you know your environment supports workers.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, num_classes, class_names

def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch function."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    """Single validation epoch function."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def test_and_report(model, test_loader, class_names, num_selected):
    """Runs the final test and prints all required metrics."""

    print("\n\n🧪 TESTING PYCCEA OPTIMIZED MODEL...")

    try:
        # Load the best weights saved during training (max validation accuracy)
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("🚨 Warning: Best model weights not found. Using final epoch weights.")

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metric Calculation ---
    accuracy = accuracy_score(all_labels, all_preds)
    total = len(all_labels)
    correct = accuracy * total

    print("\n" + "="*50)
    print(f"🎉 FINAL RESULTS ON TEST SET 🎉")
    print("="*50)
    print(f"✅ TEST ACCURACY: {accuracy:.4f} (Target: 0.99+)")
    print(f"📊 Correct: {int(correct)}/{total}")

    # PyCCEA Feature Selection Metrics
    total_features = NUM_FEATURES
    reduction_ratio = (1 - num_selected / total_features) * 100
    print(f"🎯 PyCCEA selected {num_selected}/{total_features} features (Reduction: {reduction_ratio:.2f}%)")
    print("="*50)

    # Classification Report
    if len(class_names) > 1:
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

        # Confusion Matrix (Plot)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    else:
        print("\n--- CLASSIFICATION REPORT ---")
        print("Skipping detailed report: Only one class detected.")

# =============================================================================
# 3. Main Execution
# =============================================================================

def main():

    # Map menu choices to datasets found under ./data
    datasets_map = {'1': 'aid', '2': 'uc_merced', '3': 'lc25000'}

    print("--- Dataset Selection ---")
    print("1: AID Dataset (data/AID)")
    print("2: UC Merced Land Use Dataset (data/UCMerced_LandUse)")
    print("3: LC25000 Dataset (data/lung_colon_image_set)")

    choice = input("Please select a dataset (1/2/3): ").strip()

    if choice not in datasets_map:
        print("Invalid choice. Exiting.")
        return

    selected_dataset = datasets_map[choice]

    # 1. Prepare Data from local ./data
    data_root = get_local_data_root(selected_dataset)
    if not data_root:
        return

    # 2. Create DataLoaders
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(data_root, selected_dataset)
    if train_loader is None: # Check for error from get_data_loaders
         return

    # If only 1 class is detected, stop immediately with an informative message
    if num_classes <= 1:
         print("\n\n🚨 CRITICAL ERROR: Only 1 or 0 classes detected. Fix the data path/structure for this dataset.")
         print("The path chosen was likely pointing to a single folder containing all data, not the class folders.")
         return


    # 3. Initialize PyCCEA Feature Selector
    feature_selector = PyCCEAFeatureSelector()

    # 4. Load Base Model and Run PyCCEA (ResNet-101 for max performance)
    print("\n💡 Using ResNet-101 for best chance at 99%+ accuracy.")
    base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

    # Get the feature extractor for PyCCEA
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1]).to(DEVICE)

    # Run PyCCEA on the ResNet-101 features
    selected_mask, num_selected = feature_selector.select_features(
        train_loader, DEVICE, feature_extractor, num_features=NUM_FEATURES
    )

    # 5. Initialize Feature-Selected Model
    model = FeatureSelectedResNet(base_model, num_classes, feature_selector)
    model.update_classifier(num_selected)
    model = model.to(DEVICE)

    # --- TRAINING STAGE 1: Train Classifier Only ---
    print("\n" + "="*50)
    print(f"STAGE 1: Training Classifier on {num_selected} PyCCEA Features ({STAGE1_EPOCHS} Epochs)")
    print("="*50)

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_STAGE1)
    best_val_acc = 0.0

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, nn.CrossEntropyLoss(), optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, nn.CrossEntropyLoss(), DEVICE)

        print(f"S1 Epoch {epoch+1}/{STAGE1_EPOCHS}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("⭐ Best model saved.")

    # --- TRAINING STAGE 2: Fine-Tuning ---
    print("\n" + "="*50)
    print(f"STAGE 2: Fine-Tuning All Layers ({STAGE2_EPOCHS} Epochs) - Max Accuracy Attempt")
    print("="*50)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2)

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, nn.CrossEntropyLoss(), optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, nn.CrossEntropyLoss(), DEVICE)

        print(f"S2 Epoch {epoch+1}/{STAGE2_EPOCHS}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("⭐ Best model saved.")

    # 6. Test and Report Metrics (run once after training)
    test_and_report(model, test_loader, class_names, num_selected)


if __name__ == '__main__':
    main()