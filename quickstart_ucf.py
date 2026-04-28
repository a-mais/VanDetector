"""
Script Quick Start para pipeline UCF Crime.
Execute para preparar dataset, extrair features e treinar MLP.
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Executa comando e trata erros."""
    print(f"\n{'='*60}")
    print(f"📍 {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Erro em: {description}")
        sys.exit(1)
    
    return result.returncode == 0


def main():
    print("\n")
    print("    🚀 UCF CRIME DETECTION - QUICK START")
    print("    " + "="*50)
    print("    Pipeline completo: Dataset → CNN Features → MLP Treinamento")
    print("\n")
    
    # Verificar dependências
    print("✓ Verificando dependências...")
    try:
        import torch
        import torchvision
        import sklearn
        import streamlit
        print(f"  ✅ PyTorch: {torch.__version__}")
        print(f"  ✅ TorchVision: {torchvision.__version__}")
        print(f"  ✅ scikit-learn: {sklearn.__version__}")
        print(f"  ✅ Streamlit: {streamlit.__version__}")
    except ImportError as e:
        print(f"  ❌ Falta de dependência: {e}")
        print("\n  Instale com: pip install -r requirements.txt")
        sys.exit(1)
    
    # Verificar dataset
    print("\n✓ Verificando dataset...")
    dataset_path = Path("datasets/ucfcrimedataset")
    if not dataset_path.exists():
        print(f"  ❌ Dataset não encontrado em: {dataset_path}")
        print("  Extraia o arquivo ZIP do UCF Crime Dataset nessa pasta")
        sys.exit(1)
    
    train_path = dataset_path / "Train"
    test_path = dataset_path / "Test"
    
    train_categories = len(list(train_path.glob("*")))
    test_categories = len(list(test_path.glob("*")))
    
    print(f"  ✅ Train set: {train_categories} categorias")
    print(f"  ✅ Test set: {test_categories} categorias")
    
    # Criar pasta de modelos
    Path("models").mkdir(exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PIPELINE DE TREINAMENTO")
    print("="*60)
    
    # Step 1: Preparar dataset
    if not Path("ucf_crime_dataset.csv").exists():
        run_command(
            "python prepare_ucf_dataset.py",
            "PASSO 1/3: Preparando Dataset"
        )
    else:
        print("\n✅ Dataset CSV já existe, pulando...")
    
    # Step 2: Extrair features
    features_train = Path("models/ucf_cnn_features_train.pkl")
    features_test = Path("models/ucf_cnn_features_test.pkl")
    
    if not (features_train.exists() and features_test.exists()):
        print("\n⚠️  AVISO: Este passo pode levar 4-6 horas em GPU ou 24h em CPU!")
        response = input("Deseja continuar? (s/n): ").lower().strip()
        if response != 's':
            print("Pulando extração de features...")
        else:
            run_command(
                "python extract_ucf_features.py",
                "PASSO 2/3: Extraindo Features (CNN ResNet50)"
            )
    else:
        print("\n✅ Features já extraídas, pulando...")
    
    # Step 3: Treinar MLP
    if not Path("models/ucf_crime_mlp.pkl").exists():
        run_command(
            "python train_ucf_mlp.py",
            "PASSO 3/3: Treinando MLP"
        )
    else:
        print("\n✅ MLP já treinado, pulando...")
    
    # Success
    print("\n" + "="*60)
    print("✨ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)
    
    print("\n📦 Arquivos Gerados:")
    print(f"  ✓ {Path('ucf_crime_dataset.csv').relative_to('.')}")
    print(f"  ✓ {features_train.relative_to('.')}")
    print(f"  ✓ {features_test.relative_to('.')}")
    print(f"  ✓ {Path('models/ucf_crime_mlp.pkl').relative_to('.')}")
    print(f"  ✓ {Path('models/ucf_scaler.pkl').relative_to('.')}")
    print(f"  ✓ {Path('models/ucf_mlp_report.txt').relative_to('.')}")
    print(f"  ✓ {Path('models/ucf_mlp_evaluation.png').relative_to('.')}")
    
    print("\n🚀 Próximas Ações:")
    print(f"  1. Ver relatório de treinamento:")
    print(f"     cat models/ucf_mlp_report.txt")
    print(f"")
    print(f"  2. Abrir interface web:")
    print(f"     streamlit run app_ucf.py")
    print(f"")
    print(f"  3. Testar com imagem/vídeo via UI")
    print("\n")


if __name__ == "__main__":
    main()
