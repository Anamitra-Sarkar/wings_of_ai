"""
RAG Engine - Knowledge Retrieval Module for Geo-Reason.

This module implements Retrieve-Augmented Generation (RAG) using 
documentation from QGIS/GDAL/GRASS to inform the LLM about tool usage.

Features:
- Document loading from multiple formats (PDF, Markdown, TXT)
- Text chunking with semantic overlap
- Vector store management with ChromaDB
- Efficient retrieval based on user queries
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

# Type hints for optional dependencies
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. RAG features will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Using fallback embeddings.")


class DocumentLoader:
    """
    Load and parse documents from various formats.
    
    Supports PDF, Markdown, TXT, and DOCX files for building
    the knowledge base.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.txt', '.docx', '.rst'}
    
    def __init__(self, docs_path: Union[str, Path]):
        """
        Initialize the document loader.
        
        Args:
            docs_path: Path to the documents directory
        """
        self.docs_path = Path(docs_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DocumentLoader with path: {self.docs_path}")
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Load all supported documents from the docs directory.
        
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in self.docs_path.rglob(f"*{ext}"):
                try:
                    doc = self._load_document(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _load_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single document based on its extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document dictionary or None if loading fails
        """
        extension = file_path.suffix.lower()
        
        loaders = {
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.rst': self._load_text,
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
        }
        
        loader = loaders.get(extension)
        if loader:
            content = loader(file_path)
            if content:
                return {
                    'content': content,
                    'source': str(file_path),
                    'filename': file_path.name,
                    'extension': extension,
                    'doc_id': self._generate_doc_id(file_path)
                }
        
        return None
    
    def _load_text(self, file_path: Path) -> Optional[str]:
        """Load plain text file."""
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return None
    
    def _load_markdown(self, file_path: Path) -> Optional[str]:
        """Load markdown file."""
        return self._load_text(file_path)
    
    def _load_pdf(self, file_path: Path) -> Optional[str]:
        """Load PDF file using pypdf."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return '\n\n'.join(text_parts)
        except ImportError:
            logger.warning("pypdf not installed. Skipping PDF files.")
            return None
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return None
    
    def _load_docx(self, file_path: Path) -> Optional[str]:
        """Load DOCX file using python-docx."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            return '\n\n'.join(paragraphs)
        except ImportError:
            logger.warning("python-docx not installed. Skipping DOCX files.")
            return None
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return None
    
    @staticmethod
    def _generate_doc_id(file_path: Path) -> str:
        """Generate unique document ID based on file path."""
        return hashlib.md5(str(file_path).encode()).hexdigest()[:16]


class TextChunker:
    """
    Split documents into overlapping chunks for embedding.
    
    Uses semantic chunking with configurable size and overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks with metadata.
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            List of chunk dictionaries
        """
        content = document['content']
        chunks = []
        
        # Split by paragraphs first for semantic coherence
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, document, len(chunks)))
                
                # Keep overlap
                overlap_text = '\n\n'.join(current_chunk[-2:]) if len(current_chunk) > 1 else current_chunk[-1]
                current_chunk = [overlap_text] if len(overlap_text) < self.chunk_overlap else []
                current_size = sum(len(c) for c in current_chunk)
            
            current_chunk.append(para)
            current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, document, len(chunks)))
        
        return chunks
    
    def _create_chunk(
        self, 
        text: str, 
        document: Dict[str, Any], 
        chunk_index: int
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            'text': text,
            'metadata': {
                'source': document['source'],
                'filename': document['filename'],
                'doc_id': document['doc_id'],
                'chunk_index': chunk_index,
                'chunk_id': f"{document['doc_id']}_chunk_{chunk_index}"
            }
        }


class EmbeddingProvider:
    """
    Provide embeddings for text using sentence-transformers.
    
    Falls back to simple TF-IDF if sentence-transformers is not available.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                self.model = None
        else:
            logger.warning("Using fallback embedding (TF-IDF based)")
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        else:
            return self._fallback_embed(texts)
    
    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Fallback embedding using simple hashing.
        
        This is a basic fallback for testing/development when sentence-transformers
        is not available. It uses SHA-256 for non-cryptographic hashing purposes
        to generate pseudo-random but deterministic embedding values.
        
        WARNING: This produces low-quality embeddings suitable only for
        basic functionality testing. Production use requires sentence-transformers.
        """
        import hashlib
        
        embeddings = []
        for text in texts:
            # Simple hash-based embedding (384 dimensions to match MiniLM)
            words = text.lower().split()
            embedding = [0.0] * 384
            
            for i, word in enumerate(words[:384]):
                # Use SHA-256 for better distribution than MD5
                # This is non-cryptographic use - just for deterministic pseudo-random values
                hash_val = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
                embedding[i % 384] += (hash_val / 2**32) - 0.5
            
            # Normalize
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings


class RAGEngine:
    """
    Main RAG (Retrieve-Augmented Generation) Engine.
    
    Manages document ingestion, vector storage, and retrieval
    for enhancing LLM responses with relevant GIS documentation.
    """
    
    def __init__(
        self,
        docs_path: Union[str, Path] = "./data/docs",
        vector_store_path: Union[str, Path] = "./data/vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "geo_reason_docs"
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            docs_path: Path to documentation files
            vector_store_path: Path to store ChromaDB data
            embedding_model: Name of the embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            collection_name: Name of the ChromaDB collection
        """
        self.docs_path = Path(docs_path)
        self.vector_store_path = Path(vector_store_path)
        self.collection_name = collection_name
        
        # Ensure directories exist
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.document_loader = DocumentLoader(self.docs_path)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_provider = EmbeddingProvider(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self._initialize_vector_store()
        
        logger.info("RAG Engine initialized successfully")
    
    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB vector store."""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Retrieval will be limited.")
            return
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_store_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "GeoAI documentation for RAG"}
            )
            
            logger.info(f"Vector store initialized with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
    
    def ingest_documents(self, force_reload: bool = False) -> int:
        """
        Ingest all documents from the docs directory.
        
        Args:
            force_reload: If True, clear existing documents and reload
            
        Returns:
            Number of chunks ingested
        """
        if not self.collection:
            logger.error("Vector store not initialized")
            return 0
        
        if force_reload:
            # Clear existing documents
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "GeoAI documentation for RAG"}
            )
        
        # Load documents
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            logger.warning("No documents found to ingest")
            return 0
        
        # Chunk and embed
        total_chunks = 0
        batch_size = 100
        
        ids_batch = []
        texts_batch = []
        metadatas_batch = []
        
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            
            for chunk in chunks:
                ids_batch.append(chunk['metadata']['chunk_id'])
                texts_batch.append(chunk['text'])
                metadatas_batch.append(chunk['metadata'])
                
                if len(ids_batch) >= batch_size:
                    self._add_batch(ids_batch, texts_batch, metadatas_batch)
                    total_chunks += len(ids_batch)
                    ids_batch, texts_batch, metadatas_batch = [], [], []
        
        # Add remaining
        if ids_batch:
            self._add_batch(ids_batch, texts_batch, metadatas_batch)
            total_chunks += len(ids_batch)
        
        logger.info(f"Ingested {total_chunks} chunks from {len(documents)} documents")
        return total_chunks
    
    def _add_batch(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict]
    ) -> None:
        """Add a batch of documents to the vector store."""
        embeddings = self.embedding_provider.embed(texts)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.collection:
            logger.warning("Vector store not initialized. Returning empty results.")
            return []
        
        try:
            query_embedding = self.embedding_provider.embed(query)[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            documents = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            logger.debug(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def get_gis_tool_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant GIS tool documentation for a query.
        
        Args:
            query: The user's geospatial analysis query
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Formatted context string for the LLM
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('filename', 'Unknown')
            content = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
            context_parts.append(f"### Reference {i} (Source: {source})\n{content}")
        
        return "\n\n".join(context_parts)
    
    def add_gis_documentation(self) -> None:
        """
        Add default GIS documentation to the knowledge base.
        
        This creates basic documentation for common GIS operations
        when no external documentation is available.
        """
        default_docs = self._get_default_gis_docs()
        
        for doc_name, content in default_docs.items():
            doc_path = self.docs_path / doc_name
            doc_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Added {len(default_docs)} default GIS documentation files")
        self.ingest_documents()
    
    def _get_default_gis_docs(self) -> Dict[str, str]:
        """Get default GIS documentation content."""
        return {
            "geopandas_basics.md": """# GeoPandas Basics

## Overview
GeoPandas is a Python library for working with geospatial data.

## Loading Data
```python
import geopandas as gpd

# Read from file
gdf = gpd.read_file("path/to/file.geojson")

# Read specific layer from GeoPackage
gdf = gpd.read_file("path/to/file.gpkg", layer="layer_name")
```

## CRS Handling
```python
# Check CRS
print(gdf.crs)

# Transform CRS
gdf_transformed = gdf.to_crs("EPSG:4326")

# Common CRS codes:
# EPSG:4326 - WGS84 (lat/lon)
# EPSG:3857 - Web Mercator
# EPSG:32643 - UTM Zone 43N (India)
```

## Spatial Operations
```python
# Buffer
buffered = gdf.buffer(1000)  # 1000 units in CRS

# Intersection
intersection = gpd.overlay(gdf1, gdf2, how='intersection')

# Union
union = gpd.overlay(gdf1, gdf2, how='union')

# Spatial join
joined = gpd.sjoin(gdf1, gdf2, predicate='intersects')
```
""",
            "rasterio_guide.md": """# Rasterio Guide

## Overview
Rasterio is for reading and writing geospatial raster data.

## Reading Raster Data
```python
import rasterio
import numpy as np

with rasterio.open("path/to/raster.tif") as src:
    # Read data
    data = src.read(1)  # First band
    
    # Get metadata
    crs = src.crs
    transform = src.transform
    bounds = src.bounds
```

## Writing Raster Data
```python
with rasterio.open(
    "output.tif", "w",
    driver="GTiff",
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform
) as dst:
    dst.write(data, 1)
```

## Zonal Statistics
```python
from rasterstats import zonal_stats

stats = zonal_stats(
    vectors="zones.geojson",
    raster="values.tif",
    stats=['mean', 'min', 'max', 'sum']
)
```
""",
            "osm_data.md": """# OpenStreetMap Data Access

## Using OSMnx
```python
import osmnx as ox

# Fetch buildings in an area
buildings = ox.features_from_place(
    "Kerala, India",
    tags={"building": True}
)

# Fetch by bounding box
water = ox.features_from_bbox(
    bbox=(8.18, 74.52, 12.79, 77.42),
    tags={"natural": "water"}
)

# Fetch road network
G = ox.graph_from_place("Kochi, Kerala, India")
```

## Common OSM Tags
- building: True (all buildings)
- natural: water, wood, grassland
- waterway: river, stream, canal
- landuse: residential, commercial, industrial
- highway: primary, secondary, tertiary
""",
            "flood_analysis.md": """# Flood Risk Analysis Guide

## Key Factors
1. **Elevation**: Low-lying areas are more prone to flooding
2. **Proximity to Water**: Areas near rivers, lakes, streams
3. **Drainage**: Poor drainage increases flood risk
4. **Land Use**: Impervious surfaces increase runoff

## Buffer Analysis
```python
# Create flood risk zones based on distance to water
# Use projected CRS for accurate distance calculation

# Convert to UTM for Kerala (Zone 43N)
water_utm = water_bodies.to_crs("EPSG:32643")

# Create buffers
high_risk = water_utm.buffer(100)   # Within 100m
medium_risk = water_utm.buffer(500)  # Within 500m
low_risk = water_utm.buffer(1000)    # Within 1km
```

## DEM Analysis
```python
# Identify low-lying areas from DEM
import rasterio
import numpy as np

with rasterio.open("dem.tif") as src:
    elevation = src.read(1)
    
    # Areas below threshold elevation
    low_areas = elevation < threshold
```
"""
        }


def create_rag_engine(
    docs_path: str = "./data/docs",
    vector_store_path: str = "./data/vector_store"
) -> RAGEngine:
    """
    Factory function to create and initialize a RAG Engine.
    
    Args:
        docs_path: Path to documentation files
        vector_store_path: Path to vector store
        
    Returns:
        Initialized RAG Engine instance
    """
    engine = RAGEngine(
        docs_path=docs_path,
        vector_store_path=vector_store_path
    )
    
    # Add default documentation if no docs exist
    if not list(Path(docs_path).glob("*.*")):
        engine.add_gis_documentation()
    else:
        engine.ingest_documents()
    
    return engine
