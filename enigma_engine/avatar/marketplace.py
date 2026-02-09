"""
Avatar Marketplace for Enigma AI Engine

Browse and share avatars.

Features:
- Avatar catalog
- Upload/download
- Rating system
- Categories/tags
- Preview generation

Usage:
    from enigma_engine.avatar.marketplace import AvatarMarketplace
    
    marketplace = AvatarMarketplace()
    
    # Browse
    avatars = marketplace.search("anime female")
    
    # Download
    marketplace.download(avatar_id, save_path)
"""

import hashlib
import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AvatarCategory(Enum):
    """Avatar categories."""
    ANIME = "anime"
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    CARTOON = "cartoon"
    FANTASY = "fantasy"
    SCIFI = "scifi"
    ANIMAL = "animal"
    ROBOT = "robot"
    CUSTOM = "custom"


class AvatarFormat(Enum):
    """Avatar file formats."""
    VRM = "vrm"
    GLB = "glb"
    FBX = "fbx"
    PNG = "png"  # 2D avatar
    GIF = "gif"  # Animated 2D


@dataclass
class AvatarLicense:
    """Avatar usage license."""
    name: str
    commercial_use: bool = False
    modification: bool = True
    redistribution: bool = True
    credit_required: bool = True
    url: str = ""


@dataclass
class AvatarMetadata:
    """Metadata for an avatar."""
    id: str
    name: str
    description: str = ""
    author: str = "Unknown"
    version: str = "1.0"
    category: AvatarCategory = AvatarCategory.CUSTOM
    format: AvatarFormat = AvatarFormat.VRM
    tags: List[str] = field(default_factory=list)
    thumbnail_url: str = ""
    download_url: str = ""
    file_size_bytes: int = 0
    poly_count: int = 0
    bone_count: int = 0
    blend_shapes: int = 0
    created_at: str = ""
    updated_at: str = ""
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    license: Optional[AvatarLicense] = None


@dataclass
class AvatarReview:
    """User review of an avatar."""
    avatar_id: str
    user_id: str
    rating: int  # 1-5
    comment: str = ""
    created_at: str = ""


class AvatarMarketplace:
    """Avatar marketplace for browsing and sharing."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        api_url: str = "https://api.example.com/avatars"
    ):
        """
        Initialize avatar marketplace.
        
        Args:
            cache_dir: Local cache directory
            api_url: Marketplace API endpoint
        """
        self.cache_dir = Path(cache_dir or "~/.enigma/avatar_cache").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_url = api_url
        
        # Local catalog
        self._catalog: Dict[str, AvatarMetadata] = {}
        
        # Load cached catalog
        self._load_catalog()
        
        logger.info(f"AvatarMarketplace initialized: {len(self._catalog)} cached avatars")
    
    def search(
        self,
        query: str = "",
        category: Optional[AvatarCategory] = None,
        format: Optional[AvatarFormat] = None,
        tags: Optional[List[str]] = None,
        min_rating: float = 0.0,
        limit: int = 50,
        offset: int = 0
    ) -> List[AvatarMetadata]:
        """
        Search for avatars.
        
        Args:
            query: Search query
            category: Filter by category
            format: Filter by format
            tags: Filter by tags
            min_rating: Minimum rating
            limit: Max results
            offset: Pagination offset
            
        Returns:
            List of matching avatars
        """
        results = []
        query_lower = query.lower()
        
        for avatar in self._catalog.values():
            # Apply filters
            if query_lower:
                if query_lower not in avatar.name.lower() and \
                   query_lower not in avatar.description.lower() and \
                   not any(query_lower in tag.lower() for tag in avatar.tags):
                    continue
            
            if category and avatar.category != category:
                continue
            
            if format and avatar.format != format:
                continue
            
            if tags and not any(tag in avatar.tags for tag in tags):
                continue
            
            if avatar.rating < min_rating:
                continue
            
            results.append(avatar)
        
        # Sort by rating, then downloads
        results.sort(key=lambda a: (a.rating, a.downloads), reverse=True)
        
        # Paginate
        return results[offset:offset + limit]
    
    def get_avatar(self, avatar_id: str) -> Optional[AvatarMetadata]:
        """Get avatar metadata by ID."""
        return self._catalog.get(avatar_id)
    
    def download(
        self,
        avatar_id: str,
        save_path: str,
        callback: Optional[callable] = None
    ) -> bool:
        """
        Download an avatar.
        
        Args:
            avatar_id: Avatar ID
            save_path: Path to save
            callback: Progress callback(bytes_downloaded, total_bytes)
            
        Returns:
            True if successful
        """
        avatar = self._catalog.get(avatar_id)
        if not avatar:
            logger.error(f"Avatar not found: {avatar_id}")
            return False
        
        # Check cache
        cache_path = self._get_cache_path(avatar_id)
        if cache_path.exists():
            shutil.copy(cache_path, save_path)
            logger.info(f"Copied from cache: {avatar_id}")
            return True
        
        # Download (mock implementation)
        try:
            logger.info(f"Downloading avatar: {avatar.name}")
            
            # In real implementation, use requests/aiohttp
            # Here we simulate with local file if available
            if avatar.download_url.startswith("file://"):
                source = avatar.download_url[7:]
                if Path(source).exists():
                    shutil.copy(source, save_path)
                    shutil.copy(source, cache_path)
                    return True
            
            logger.warning(f"Download URL not accessible: {avatar.download_url}")
            return False
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def upload(
        self,
        file_path: str,
        metadata: AvatarMetadata,
        callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Upload an avatar to marketplace.
        
        Args:
            file_path: Path to avatar file
            metadata: Avatar metadata
            callback: Progress callback
            
        Returns:
            Avatar ID if successful
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Generate ID
        import uuid
        avatar_id = str(uuid.uuid4())
        
        # Set metadata
        metadata.id = avatar_id
        metadata.file_size_bytes = file_path.stat().st_size
        metadata.created_at = datetime.now().isoformat()
        metadata.updated_at = metadata.created_at
        metadata.download_url = f"file://{file_path.absolute()}"
        
        # Detect format
        suffix = file_path.suffix.lower()
        format_map = {'.vrm': AvatarFormat.VRM, '.glb': AvatarFormat.GLB, 
                     '.fbx': AvatarFormat.FBX, '.png': AvatarFormat.PNG, '.gif': AvatarFormat.GIF}
        metadata.format = format_map.get(suffix, AvatarFormat.VRM)
        
        # Add to catalog
        self._catalog[avatar_id] = metadata
        self._save_catalog()
        
        # Copy to cache
        cache_path = self._get_cache_path(avatar_id)
        shutil.copy(file_path, cache_path)
        
        logger.info(f"Uploaded avatar: {metadata.name} ({avatar_id})")
        return avatar_id
    
    def rate(
        self,
        avatar_id: str,
        rating: int,
        user_id: str = "anonymous",
        comment: str = ""
    ) -> bool:
        """
        Rate an avatar.
        
        Args:
            avatar_id: Avatar ID
            rating: Rating 1-5
            user_id: User ID
            comment: Optional comment
            
        Returns:
            True if successful
        """
        avatar = self._catalog.get(avatar_id)
        if not avatar:
            return False
        
        rating = max(1, min(5, rating))
        
        # Update average rating
        total = avatar.rating * avatar.rating_count + rating
        avatar.rating_count += 1
        avatar.rating = total / avatar.rating_count
        
        # Save review
        review = AvatarReview(
            avatar_id=avatar_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            created_at=datetime.now().isoformat()
        )
        
        reviews_file = self.cache_dir / f"{avatar_id}_reviews.json"
        reviews = []
        if reviews_file.exists():
            reviews = json.loads(reviews_file.read_text())
        reviews.append({
            "user_id": review.user_id,
            "rating": review.rating,
            "comment": review.comment,
            "created_at": review.created_at
        })
        reviews_file.write_text(json.dumps(reviews, indent=2))
        
        self._save_catalog()
        return True
    
    def get_reviews(self, avatar_id: str) -> List[AvatarReview]:
        """Get reviews for an avatar."""
        reviews_file = self.cache_dir / f"{avatar_id}_reviews.json"
        if not reviews_file.exists():
            return []
        
        data = json.loads(reviews_file.read_text())
        return [AvatarReview(avatar_id=avatar_id, **r) for r in data]
    
    def get_featured(self, limit: int = 10) -> List[AvatarMetadata]:
        """Get featured/top avatars."""
        return self.search(min_rating=4.0, limit=limit)
    
    def get_recent(self, limit: int = 10) -> List[AvatarMetadata]:
        """Get recently added avatars."""
        recent = sorted(
            self._catalog.values(),
            key=lambda a: a.created_at or "",
            reverse=True
        )
        return recent[:limit]
    
    def get_categories(self) -> Dict[AvatarCategory, int]:
        """Get category counts."""
        counts = {cat: 0 for cat in AvatarCategory}
        for avatar in self._catalog.values():
            counts[avatar.category] += 1
        return counts
    
    def get_popular_tags(self, limit: int = 20) -> List[tuple]:
        """Get popular tags with counts."""
        tag_counts = {}
        for avatar in self._catalog.values():
            for tag in avatar.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:limit]
    
    def export_avatar_pack(
        self,
        avatar_ids: List[str],
        output_path: str
    ) -> bool:
        """Export multiple avatars as a pack."""
        output_path = Path(output_path)
        
        with zipfile.ZipFile(output_path, 'w') as zf:
            manifest = {"avatars": []}
            
            for avatar_id in avatar_ids:
                avatar = self._catalog.get(avatar_id)
                if not avatar:
                    continue
                
                cache_path = self._get_cache_path(avatar_id)
                if cache_path.exists():
                    zf.write(cache_path, f"avatars/{avatar_id}{cache_path.suffix}")
                    manifest["avatars"].append({
                        "id": avatar.id,
                        "name": avatar.name,
                        "category": avatar.category.value,
                        "format": avatar.format.value
                    })
            
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        
        return True
    
    def import_avatar_pack(self, pack_path: str) -> List[str]:
        """Import an avatar pack."""
        pack_path = Path(pack_path)
        imported = []
        
        with zipfile.ZipFile(pack_path, 'r') as zf:
            manifest_data = zf.read("manifest.json")
            manifest = json.loads(manifest_data)
            
            for avatar_info in manifest.get("avatars", []):
                avatar_id = avatar_info["id"]
                
                # Extract avatar file
                for name in zf.namelist():
                    if name.startswith(f"avatars/{avatar_id}"):
                        ext = Path(name).suffix
                        cache_path = self._get_cache_path(avatar_id)
                        cache_path = cache_path.with_suffix(ext)
                        
                        with zf.open(name) as src:
                            cache_path.write_bytes(src.read())
                        
                        # Create metadata
                        metadata = AvatarMetadata(
                            id=avatar_id,
                            name=avatar_info.get("name", avatar_id),
                            category=AvatarCategory(avatar_info.get("category", "custom")),
                            format=AvatarFormat(avatar_info.get("format", "vrm")),
                            download_url=f"file://{cache_path}"
                        )
                        
                        self._catalog[avatar_id] = metadata
                        imported.append(avatar_id)
        
        self._save_catalog()
        return imported
    
    def _get_cache_path(self, avatar_id: str) -> Path:
        """Get cache path for avatar."""
        avatar = self._catalog.get(avatar_id)
        ext = f".{avatar.format.value}" if avatar else ".vrm"
        return self.cache_dir / f"{avatar_id}{ext}"
    
    def _load_catalog(self):
        """Load local catalog."""
        catalog_file = self.cache_dir / "catalog.json"
        if catalog_file.exists():
            try:
                data = json.loads(catalog_file.read_text())
                for item in data:
                    item['category'] = AvatarCategory(item.get('category', 'custom'))
                    item['format'] = AvatarFormat(item.get('format', 'vrm'))
                    if item.get('license'):
                        item['license'] = AvatarLicense(**item['license'])
                    self._catalog[item['id']] = AvatarMetadata(**item)
            except Exception as e:
                logger.error(f"Failed to load catalog: {e}")
    
    def _save_catalog(self):
        """Save local catalog."""
        catalog_file = self.cache_dir / "catalog.json"
        data = []
        for avatar in self._catalog.values():
            item = {
                'id': avatar.id,
                'name': avatar.name,
                'description': avatar.description,
                'author': avatar.author,
                'version': avatar.version,
                'category': avatar.category.value,
                'format': avatar.format.value,
                'tags': avatar.tags,
                'thumbnail_url': avatar.thumbnail_url,
                'download_url': avatar.download_url,
                'file_size_bytes': avatar.file_size_bytes,
                'poly_count': avatar.poly_count,
                'bone_count': avatar.bone_count,
                'blend_shapes': avatar.blend_shapes,
                'created_at': avatar.created_at,
                'updated_at': avatar.updated_at,
                'downloads': avatar.downloads,
                'rating': avatar.rating,
                'rating_count': avatar.rating_count,
            }
            if avatar.license:
                item['license'] = {
                    'name': avatar.license.name,
                    'commercial_use': avatar.license.commercial_use,
                    'modification': avatar.license.modification,
                    'redistribution': avatar.license.redistribution,
                    'credit_required': avatar.license.credit_required,
                    'url': avatar.license.url
                }
            data.append(item)
        catalog_file.write_text(json.dumps(data, indent=2))


# Sample avatars for testing
def create_sample_catalog(marketplace: AvatarMarketplace):
    """Create sample avatars for testing."""
    samples = [
        AvatarMetadata(
            id="sample_001",
            name="Default Anime Avatar",
            description="A friendly anime-style avatar",
            author="Enigma AI",
            category=AvatarCategory.ANIME,
            format=AvatarFormat.VRM,
            tags=["anime", "female", "default"],
            rating=4.5,
            rating_count=10
        ),
        AvatarMetadata(
            id="sample_002",
            name="Robot Assistant",
            description="A helpful robot avatar",
            author="Enigma AI",
            category=AvatarCategory.ROBOT,
            format=AvatarFormat.GLB,
            tags=["robot", "assistant", "scifi"],
            rating=4.2,
            rating_count=5
        )
    ]
    
    for sample in samples:
        sample.created_at = datetime.now().isoformat()
        marketplace._catalog[sample.id] = sample
    
    marketplace._save_catalog()
