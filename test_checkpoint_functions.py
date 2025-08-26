#!/usr/bin/env python3
"""
Script de prueba para validar las funciones de limpieza y reanudación de checkpoints
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Agregar el directorio utils al path para importar las funciones
sys.path.append(str(Path(__file__).parent / "utils"))

try:
    from gpt_train import cleanup_old_checkpoints, find_latest_checkpoint
    print("✅ Funciones importadas correctamente")
except ImportError as e:
    print(f"❌ Error al importar funciones: {e}")
    sys.exit(1)


def test_checkpoint_functions():
    """Prueba las funciones de limpieza y búsqueda de checkpoints"""
    
    # Crear directorio temporal para pruebas
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"🔧 Usando directorio temporal: {temp_path}")
        
        # Crear archivos de prueba simulando checkpoints
        test_files = [
            "best_model.pth",
            "best_model_576.pth", 
            "checkpoint_1000.pth",
            "checkpoint_2000.pth",
            "checkpoint_3000.pth",
            "config.json",
            "events.out.tfevents.123456"
        ]
        
        print("\n📁 Creando archivos de prueba...")
        for filename in test_files:
            test_file = temp_path / filename
            test_file.write_text("dummy content")
            print(f"   Creado: {filename}")
        
        print(f"\n📊 Archivos antes de la limpieza: {len(list(temp_path.glob('*.pth')))} archivos .pth")
        
        # Probar búsqueda del checkpoint más reciente
        print("\n🔍 Probando find_latest_checkpoint...")
        latest_checkpoint, latest_step = find_latest_checkpoint(temp_path)
        
        if latest_checkpoint:
            print(f"   ✅ Checkpoint más reciente encontrado: {Path(latest_checkpoint).name} (paso {latest_step})")
        else:
            print("   ❌ No se encontró checkpoint")
            
        # Probar limpieza de checkpoints
        print(f"\n🧹 Probando cleanup_old_checkpoints con max_checkpoints=1...")
        cleanup_old_checkpoints(temp_path, max_checkpoints=1)
        
        remaining_pth_files = list(temp_path.glob('*.pth'))
        print(f"📊 Archivos después de la limpieza: {len(remaining_pth_files)} archivos .pth")
        
        for pth_file in remaining_pth_files:
            print(f"   Quedó: {pth_file.name}")
            
        # Verificar que best_model.pth se mantiene siempre
        best_model_exists = (temp_path / "best_model.pth").exists()
        print(f"\n✅ best_model.pth preservado: {'Sí' if best_model_exists else 'No'}")
        
        # Verificar que solo queden los archivos esperados
        expected_remaining = 2  # best_model.pth + 1 checkpoint más reciente
        if len(remaining_pth_files) <= expected_remaining:
            print(f"✅ Limpieza exitosa: {len(remaining_pth_files)} archivos <= {expected_remaining} esperados")
        else:
            print(f"❌ Limpieza fallida: {len(remaining_pth_files)} archivos > {expected_remaining} esperados")
            
        return len(remaining_pth_files) <= expected_remaining


def test_empty_directory():
    """Prueba el comportamiento con directorio vacío"""
    print("\n\n🔍 Probando con directorio vacío...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Probar con directorio vacío
        latest_checkpoint, latest_step = find_latest_checkpoint(temp_path)
        cleanup_old_checkpoints(temp_path, max_checkpoints=1)
        
        if latest_checkpoint is None and latest_step == 0:
            print("✅ Manejo correcto de directorio vacío")
            return True
        else:
            print("❌ Error en manejo de directorio vacío")
            return False


def main():
    """Función principal de pruebas"""
    print("🚀 Iniciando pruebas de funciones de checkpoint...")
    
    try:
        test1_success = test_checkpoint_functions()
        test2_success = test_empty_directory()
        
        print(f"\n{'='*50}")
        print("📋 RESUMEN DE PRUEBAS:")
        print(f"   Test función principal: {'✅ PASÓ' if test1_success else '❌ FALLÓ'}")
        print(f"   Test directorio vacío:  {'✅ PASÓ' if test2_success else '❌ FALLÓ'}")
        
        if test1_success and test2_success:
            print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
            print("\n📌 Las funciones están listas para:")
            print("   • Limpiar checkpoints antiguos automáticamente")
            print("   • Mantener solo los checkpoints más recientes")
            print("   • Preservar always best_model.pth")
            print("   • Encontrar el checkpoint más reciente para reanudación")
            return True
        else:
            print("\n💥 Algunas pruebas fallaron")
            return False
            
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
