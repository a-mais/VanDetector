"""
Recuperação de Extração de Features - Detecta erro de disco e reexecuta com HDF5
"""

import subprocess
import sys
from pathlib import Path
import time

def check_disk_space():
    """Verifica espaço em disco."""
    import shutil
    stat = shutil.disk_usage("D:\\")
    free_gb = stat.free / 1024**3
    print(f"📊 Espaço disponível: {free_gb:.2f} GB")
    return free_gb

def resume_extraction():
    """Retoma extração com HDF5."""
    
    print("="*60)
    print("🔄 RECUPERAÇÃO DE EXTRAÇÃO DE FEATURES")
    print("="*60)
    
    # Check disk
    free_space = check_disk_space()
    
    if free_space < 10:
        print(f"\n⚠️  Espaço insuficiente: {free_space:.2f} GB < 10 GB necessários")
        print("\n💡 Opções:")
        print("   1. Limpar dataset original após extração")
        print("   2. Usar outro disco com mais espaço")
        print("   3. Processar em batches menores")
        sys.exit(1)
    
    print(f"✅ Espaço suficiente: {free_space:.2f} GB\n")
    
    # Check status
    features_train = Path("models/ucf_cnn_features_train.h5")
    features_test = Path("models/ucf_cnn_features_test.h5")
    
    if features_train.exists() and features_test.exists():
        print("✅ Features HDF5 já existem!")
        print(f"   Train: {features_train.stat().st_size / 1024**3:.2f} GB")
        print(f"   Test:  {features_test.stat().st_size / 1024**3:.2f} GB")
        return True
    
    # Run extraction
    print("🔄 Retomando extração com HDF5...")
    print("(Isso leva ~30-40 minutos em GPU)\n")
    
    result = subprocess.run(
        [sys.executable, "extract_ucf_features.py"],
        cwd=Path.cwd()
    )
    
    return result.returncode == 0


if __name__ == "__main__":
    success = resume_extraction()
    
    if success:
        print("\n" + "="*60)
        print("✨ EXTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print("\nPróximo passo: treinar MLP")
        print("$ python train_ucf_mlp.py")
    else:
        print("\n❌ Erro na extração")
        sys.exit(1)
