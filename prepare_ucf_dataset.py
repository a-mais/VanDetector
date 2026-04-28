"""
Preparação do dataset UCF Crime para classificação.
Cria um CSV com paths e labels para treino/validação.
"""

from pathlib import Path
import pandas as pd
from collections import defaultdict

# Configuração
DATASET_ROOT = Path("datasets/ucfcrimedataset")
OUTPUT_CSV = Path("ucf_crime_dataset.csv")

# Mapa de categorias: crime ou normal
CRIME_CATEGORIES = {
    "Abuse", "Arrest", "Arson", "Assault", 
    "Burglary", "Explosion", "Fighting", "Robbery", 
    "RoadAccidents", "Shooting"
}
NORMAL_CATEGORIES = {"NormalVideos"}

def prepare_dataset():
    """Cria CSV com todos os frames e seus labels."""
    
    data = []
    stats = defaultdict(int)
    
    for split in ["Train", "Test"]:
        split_path = DATASET_ROOT / split
        
        for category_dir in sorted(split_path.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            
            # Determinar label
            if category in CRIME_CATEGORIES:
                label = "crime"
            elif category in NORMAL_CATEGORIES:
                label = "normal"
            else:
                print(f"⚠️  Categoria desconhecida: {category}")
                continue
            
            # Contar frames
            frame_count = 0
            for frame_path in category_dir.glob("*.png"):
                frame_count += 1
                relative_path = str(frame_path.relative_to(DATASET_ROOT))
                data.append({
                    "path": relative_path,
                    "category": category,
                    "label": label,
                    "split": split
                })
                
                if frame_count % 50000 == 0:
                    print(f"  Processado {frame_count} frames de {category}...")
            
            stats[f"{split}_{category}"] = frame_count
            print(f"✅ {split}/{category}: {frame_count} frames")
    
    # Criar DataFrame e salvar
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"Dataset salvo em: {OUTPUT_CSV}")
    print(f"Total de frames: {len(df)}")
    print(f"Distribuição:")
    print(f"  Crime: {len(df[df['label'] == 'crime'])}")
    print(f"  Normal: {len(df[df['label'] == 'normal'])}")
    print(f"Distribuição por split:")
    print(df["split"].value_counts())
    print(f"{'='*60}\n")
    
    return df

if __name__ == "__main__":
    print("🔄 Preparando dataset UCF Crime...")
    df = prepare_dataset()
