"""
Memory Export/Import API
Allows exporting and importing memories across sessions and devices.
"""
import logging
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from .categorization import MemoryCategorization, Memory, MemoryType
from .vector_db import VectorDBInterface

logger = logging.getLogger(__name__)


class MemoryExporter:
    """Export memories to various formats."""
    
    def __init__(self, memory_system: MemoryCategorization):
        """
        Initialize exporter.
        
        Args:
            memory_system: Memory categorization system
        """
        self.memory_system = memory_system
    
    def export_to_json(
        self,
        path: Path,
        memory_types: Optional[List[MemoryType]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Export memories to JSON format.
        
        Args:
            path: Path to save JSON file
            memory_types: List of memory types to export (None = all)
            include_metadata: Include metadata in export
            
        Returns:
            Export statistics
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get memories to export
        if memory_types:
            memories = []
            for mem_type in memory_types:
                memories.extend(self.memory_system.get_memories_by_type(mem_type))
        else:
            memories = self.memory_system.get_all_memories()
        
        # Prepare export data
        export_data = {
            'version': '1.0',
            'export_date': datetime.now().isoformat(),
            'total_memories': len(memories),
            'memories': [mem.to_dict() for mem in memories]
        }
        
        if include_metadata:
            export_data['statistics'] = self.memory_system.get_statistics()
            export_data['config'] = self.memory_system.config
        
        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        stats = {
            'exported_count': len(memories),
            'file_path': str(path),
            'file_size_bytes': path.stat().st_size
        }
        
        logger.info(f"Exported {stats['exported_count']} memories to {path}")
        return stats
    
    def export_to_csv(
        self,
        path: Path,
        memory_types: Optional[List[MemoryType]] = None
    ) -> Dict[str, Any]:
        """
        Export memories to CSV format.
        
        Args:
            path: Path to save CSV file
            memory_types: List of memory types to export
            
        Returns:
            Export statistics
        """
        import csv
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get memories
        if memory_types:
            memories = []
            for mem_type in memory_types:
                memories.extend(self.memory_system.get_memories_by_type(mem_type))
        else:
            memories = self.memory_system.get_all_memories()
        
        # Write CSV
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'content', 'type', 'timestamp', 'importance', 'access_count'])
            
            for mem in memories:
                writer.writerow([
                    mem.id,
                    mem.content,
                    mem.memory_type.value,
                    mem.timestamp,
                    mem.importance,
                    mem.access_count
                ])
        
        stats = {
            'exported_count': len(memories),
            'file_path': str(path),
            'file_size_bytes': path.stat().st_size
        }
        
        logger.info(f"Exported {stats['exported_count']} memories to CSV: {path}")
        return stats
    
    def export_to_archive(
        self,
        path: Path,
        include_vectors: bool = False,
        vector_db: Optional[VectorDBInterface] = None
    ) -> Dict[str, Any]:
        """
        Export complete memory system to a zip archive.
        
        Args:
            path: Path to save archive
            include_vectors: Include vector database
            vector_db: Vector database instance (required if include_vectors=True)
            
        Returns:
            Export statistics
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for export
        temp_dir = path.parent / f"temp_export_{datetime.now().timestamp()}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Export memories to JSON
            memories_path = temp_dir / "memories.json"
            self.export_to_json(memories_path, include_metadata=True)
            
            # Export configuration
            config_path = temp_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'version': '1.0',
                    'export_date': datetime.now().isoformat(),
                    'config': self.memory_system.config,
                    'statistics': self.memory_system.get_statistics()
                }, f, indent=2)
            
            # Export vector database if requested
            if include_vectors and vector_db:
                vector_path = temp_dir / "vectors.db"
                vector_db.save(vector_path)
            
            # Create archive
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(temp_dir))
            
            stats = {
                'archive_path': str(path),
                'archive_size_bytes': path.stat().st_size,
                'includes_vectors': include_vectors,
                'export_date': datetime.now().isoformat()
            }
            
            logger.info(f"Created memory archive: {path}")
            return stats
            
        finally:
            # Clean up temporary directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


class MemoryImporter:
    """Import memories from various formats."""
    
    def __init__(self, memory_system: MemoryCategorization):
        """
        Initialize importer.
        
        Args:
            memory_system: Memory categorization system
        """
        self.memory_system = memory_system
    
    def import_from_json(
        self,
        path: Path,
        merge: bool = True,
        overwrite_duplicates: bool = False
    ) -> Dict[str, Any]:
        """
        Import memories from JSON format.
        
        Args:
            path: Path to JSON file
            merge: Merge with existing memories (vs replace)
            overwrite_duplicates: Overwrite existing memories with same ID
            
        Returns:
            Import statistics
        """
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")
        
        # Load data
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        memories = data.get('memories', [])
        
        # Clear existing if not merging
        if not merge:
            self.memory_system.clear_all()
        
        # Import memories
        imported = 0
        skipped = 0
        
        for mem_data in memories:
            mem = Memory.from_dict(mem_data)
            
            # Check for duplicates
            existing = self.memory_system.get_memory(mem.id, mem.memory_type)
            if existing and not overwrite_duplicates:
                skipped += 1
                continue
            
            # Add to memory system
            if existing:
                self.memory_system.remove_memory(mem.id, mem.memory_type)
            
            self.memory_system.categories[mem.memory_type].memories[mem.id] = mem
            imported += 1
        
        stats = {
            'imported_count': imported,
            'skipped_count': skipped,
            'total_in_file': len(memories),
            'source_file': str(path)
        }
        
        logger.info(f"Imported {imported} memories (skipped {skipped} duplicates)")
        return stats
    
    def import_from_csv(
        self,
        path: Path,
        merge: bool = True
    ) -> Dict[str, Any]:
        """
        Import memories from CSV format.
        
        Args:
            path: Path to CSV file
            merge: Merge with existing memories
            
        Returns:
            Import statistics
        """
        import csv
        
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")
        
        if not merge:
            self.memory_system.clear_all()
        
        imported = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    mem_type = MemoryType(row['type'])
                    self.memory_system.add_memory(
                        content=row['content'],
                        memory_type=mem_type,
                        importance=float(row.get('importance', 0.5)),
                        id_=row['id']
                    )
                    imported += 1
                except Exception as e:
                    logger.warning(f"Failed to import row: {e}")
        
        stats = {
            'imported_count': imported,
            'source_file': str(path)
        }
        
        logger.info(f"Imported {imported} memories from CSV")
        return stats
    
    def import_from_archive(
        self,
        path: Path,
        merge: bool = True,
        import_vectors: bool = False,
        vector_db: Optional[VectorDBInterface] = None
    ) -> Dict[str, Any]:
        """
        Import complete memory system from archive.
        
        Args:
            path: Path to archive file
            merge: Merge with existing memories
            import_vectors: Import vector database
            vector_db: Vector database instance (required if import_vectors=True)
            
        Returns:
            Import statistics
        """
        if not path.exists():
            raise FileNotFoundError(f"Archive not found: {path}")
        
        # Create temporary extraction directory
        temp_dir = path.parent / f"temp_import_{datetime.now().timestamp()}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract archive
            with zipfile.ZipFile(path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Import memories
            memories_path = temp_dir / "memories.json"
            stats = self.import_from_json(memories_path, merge)
            
            # Import vectors if requested
            if import_vectors and vector_db:
                vector_path = temp_dir / "vectors.db"
                if vector_path.exists():
                    vector_db.load(vector_path)
                    stats['vectors_imported'] = True
            
            stats['archive_path'] = str(path)
            
            logger.info(f"Imported memory archive: {path}")
            return stats
            
        finally:
            # Clean up
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def validate_import_file(self, path: Path) -> Dict[str, Any]:
        """
        Validate import file without importing.
        
        Args:
            path: Path to import file
            
        Returns:
            Validation results
        """
        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                
                return {
                    'valid': True,
                    'format': 'json',
                    'memory_count': len(data.get('memories', [])),
                    'version': data.get('version', 'unknown')
                }
            
            elif path.suffix == '.zip':
                with zipfile.ZipFile(path, 'r') as zipf:
                    file_list = zipf.namelist()
                
                return {
                    'valid': 'memories.json' in file_list,
                    'format': 'archive',
                    'contents': file_list
                }
            
            else:
                return {
                    'valid': False,
                    'error': 'Unsupported file format'
                }
        
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
