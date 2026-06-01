from pathlib import Path
import random
import shutil

random.seed()

base_dir = Path(".")
folders = [
    "arranca", "almacen", "base", "recoge", "suelo",
    "mira", "izquierda", "grua", "freno", "entrega"
]

train_dir = base_dir / "train"
test_dir = base_dir / "test"

train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

for folder_name in folders:
    source_dir = base_dir / folder_name
    train_class_dir = train_dir / folder_name
    test_class_dir = test_dir / folder_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # Regresa archivos si ya habías intentado dividir antes
    for old_file in train_class_dir.glob("*"):
        shutil.move(str(old_file), str(source_dir / old_file.name))

    for old_file in test_class_dir.glob("*"):
        shutil.move(str(old_file), str(source_dir / old_file.name))

    files = sorted([f for f in source_dir.iterdir() if f.is_file()])

    if len(files) != 60:
        print(f"WARNING: {folder_name} tiene {len(files)} archivos, no 45")

    for start in [0, 15, 30, 45]:
        group = files[start:start + 15]
        random.shuffle(group)

        train_files = group[:10]
        test_files = group[10:15]

        for file in train_files:
            shutil.move(str(file), str(train_class_dir / file.name))

        for file in test_files:
            shutil.move(str(file), str(test_class_dir / file.name))

print("Listo.")
print("Train:", len(list(train_dir.rglob("*.*"))))
print("Test:", len(list(test_dir.rglob("*.*"))))
