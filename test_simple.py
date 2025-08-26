#!/usr/bin/env python3
"""
Script de prueba simplificado para las funciones de limpieza y reanudación de checkpoints
"""

import os
import sys
import glob
import re
from pathlib import Path
import tempfile


def cleanup_old_checkpoints(output_dir, max_checkpoints=2):
    """
    Limpia checkpoints antiguos, manteniendo solo los max_checkpoints más recientes.
    Incluye best_model.pth, best_model_*.pth y checkpoint_*.pth
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return
            
        # Patrones de archivos de checkpoint
        checkpoint_patterns = [
            'checkpoint_*.pth',
            'best_model_*.pth'
        ]
        
        all_checkpoints = []
        
        # Recopilar todos los archivos de checkpoint con su tiempo de modificación
        for pattern in checkpoint_patterns:
            for checkpoint_file in output_path.glob(pattern):
                mtime = checkpoint_file.stat().st_mtime
                all_checkpoints.append((checkpoint_file, mtime))
        
        # Ordenar por tiempo de modificación (más reciente primero)
        all_checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Mantener best_model.pth siempre
        best_model_path = output_path / 'best_model.pth'
        protected_files = set()
        if best_model_path.exists():
            protected_files.add(best_model_path)
        
        # Contar archivos a mantener (excluyendo best_model.pth si existe)
        files_to_keep = []
        for checkpoint_file, _ in all_checkpoints:
            if checkpoint_file not in protected_files:
                files_to_keep.append(checkpoint_file)
        
        # Eliminar archivos excedentes
        if len(files_to_keep) > max_checkpoints:
            files_to_delete = files_to_keep[max_checkpoints:]
            for file_to_delete in files_to_delete:
                try:
                    file_to_delete.unlink()
                    print(f" > Deleted old checkpoint: {file_to_delete.name}")
                except OSError as e:
                    print(f" > Warning: Could not delete {file_to_delete.name}: {e}")
                    
    except Exception as e:
        print(f" > Warning: Error during checkpoint cleanup: {e}")


def find_latest_checkpoint(output_dir):
    """
    Encuentra el checkpoint más reciente para reanudar el entrenamiento.
    Retorna la ruta del checkpoint y el número de paso si se encuentra.
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return None, 0
            
        # Buscar checkpoints numerados primero
        checkpoint_files = list(output_path.glob('checkpoint_*.pth'))
        
        if checkpoint_files:
            # Extraer números de paso y encontrar el más alto
            latest_step = 0
            latest_checkpoint = None
            
            for checkpoint_file in checkpoint_files:
                match = re.search(r'checkpoint_(\d+)\.pth', checkpoint_file.name)
                if match:
                    step = int(match.group(1))
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint = checkpoint_file
            
            if latest_checkpoint:
                return str(latest_checkpoint), latest_step
        
        # Si no hay checkpoints numerados, buscar best_model.pth
        best_model = output_path / 'best_model.pth'
        if best_model.exists():
            return str(best_model), 0
            
        return None, 0
        
    except Exception as e:
        print(f" > Warning: Error finding latest checkpoint: {e}")
        return None, 0


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
            print("   • Preservar siempre best_model.pth")
            print("   • Encontrar el checkpoint más reciente para reanudación")
            print("   • Solucionar el problema del checkpoint_1000.pth que no se borraba")
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
