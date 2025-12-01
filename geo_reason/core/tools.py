"""
GIS Tools Module for Geo-Reason.

Provides custom geospatial tools for the ReAct agent:
- load_vector_layer: Load vector data using GeoPandas
- buffer_geometry: Create buffer zones around geometries
- calculate_raster_stats: Calculate zonal statistics using Rasterio
- fetch_osm_data: Fetch data from OpenStreetMap
- spatial_join: Join layers based on spatial relationships
- dissolve: Dissolve/merge geometries
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

# Type hints for optional dependencies
try:
    import geopandas as gpd
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("GeoPandas not available. Vector operations will be limited.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import rasterio
    from rasterio.mask import mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("Rasterio not available. Raster operations will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class GeoTools:
    """
    Collection of geospatial analysis tools for the Geo-Reason agent.
    
    Each tool is designed to be called by the agent during the ReAct loop.
    Tools handle CRS transformations and provide detailed error messages.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize GeoTools.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("GeoTools initialized")
    
    @staticmethod
    def load_vector_layer(
        path: Union[str, Path],
        layer: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> "gpd.GeoDataFrame":
        """
        Load vector geospatial data from various formats.
        
        Args:
            path: Path to the vector file (GeoJSON, Shapefile, GeoPackage, etc.)
            layer: Layer name for multi-layer formats like GeoPackage
            bbox: Bounding box filter (minx, miny, maxx, maxy)
            
        Returns:
            GeoDataFrame with geometry and attributes
            
        Raises:
            ValueError: If the file cannot be loaded
            ImportError: If GeoPandas is not available
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for vector operations")
        
        try:
            path = Path(path)
            
            if not path.exists():
                raise ValueError(f"File not found: {path}")
            
            # Build read parameters
            read_kwargs: Dict[str, Any] = {}
            if layer:
                read_kwargs["layer"] = layer
            if bbox:
                read_kwargs["bbox"] = bbox
            
            # Read the file
            gdf = gpd.read_file(str(path), **read_kwargs)
            
            # Validate
            if gdf.empty:
                logger.warning(f"Empty GeoDataFrame loaded from {path}")
            else:
                logger.info(
                    f"Loaded {len(gdf)} features from {path.name}, "
                    f"CRS: {gdf.crs}"
                )
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading vector layer from {path}: {e}")
            raise ValueError(f"Failed to load vector layer: {e}") from e
    
    @staticmethod
    def buffer_geometry(
        gdf: "gpd.GeoDataFrame",
        distance: float,
        resolution: int = 16,
        target_crs: Optional[str] = None
    ) -> "gpd.GeoDataFrame":
        """
        Create buffer zones around geometries.
        
        For accurate distance-based buffering, the function will transform
        to a projected CRS if the input is in geographic coordinates.
        
        Args:
            gdf: Input GeoDataFrame
            distance: Buffer distance in meters (if CRS is geographic) or CRS units
            resolution: Number of segments for circular buffers
            target_crs: Optional CRS for output (defaults to input CRS)
            
        Returns:
            GeoDataFrame with buffered geometries
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for buffer operations")
        
        try:
            original_crs = gdf.crs
            work_gdf = gdf.copy()
            
            # Check if we need to project for accurate buffering
            needs_projection = original_crs and original_crs.is_geographic
            
            if needs_projection:
                # Estimate UTM zone from centroid
                centroid = work_gdf.geometry.unary_union.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                hemisphere = "N" if centroid.y >= 0 else "S"
                epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
                projected_crs = f"EPSG:{epsg}"
                
                logger.debug(f"Projecting to {projected_crs} for accurate buffering")
                work_gdf = work_gdf.to_crs(projected_crs)
            
            # Apply buffer
            work_gdf["geometry"] = work_gdf.geometry.buffer(
                distance, 
                resolution=resolution
            )
            
            # Transform back to original or target CRS
            output_crs = target_crs or original_crs
            if output_crs:
                work_gdf = work_gdf.to_crs(output_crs)
            
            logger.info(f"Created {distance}m buffer for {len(work_gdf)} features")
            return work_gdf
            
        except Exception as e:
            logger.error(f"Error creating buffer: {e}")
            raise ValueError(f"Failed to create buffer: {e}") from e
    
    @staticmethod
    def calculate_raster_stats(
        raster_path: Union[str, Path],
        vector: Union[str, Path, "gpd.GeoDataFrame"],
        stats: Optional[List[str]] = None
    ) -> "pd.DataFrame":
        """
        Calculate zonal statistics for raster data within vector boundaries.
        
        Args:
            raster_path: Path to the raster file
            vector: Path to vector file or GeoDataFrame defining zones
            stats: List of statistics to calculate (mean, min, max, sum, count, std)
            
        Returns:
            DataFrame with statistics per zone
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio is required for raster operations")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for statistics calculation")
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for DataFrame output")
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for vector operations")
        
        stats = stats or ["mean", "min", "max", "count"]
        raster_path = Path(raster_path)
        
        try:
            # Load vector if path provided
            if isinstance(vector, (str, Path)):
                zones = gpd.read_file(str(vector))
            else:
                zones = vector
            
            # Open raster
            with rasterio.open(str(raster_path)) as src:
                # Ensure CRS match
                if zones.crs != src.crs:
                    zones = zones.to_crs(src.crs)
                
                results = []
                
                for idx, row in zones.iterrows():
                    try:
                        # Mask raster with geometry
                        out_image, _ = mask(
                            src, 
                            [mapping(row.geometry)], 
                            crop=True,
                            nodata=src.nodata
                        )
                        
                        # Get valid data (exclude nodata)
                        if src.nodata is not None:
                            data = out_image[out_image != src.nodata]
                        else:
                            data = out_image.flatten()
                        
                        # Calculate statistics
                        zone_stats = {"zone_id": idx}
                        
                        if len(data) > 0:
                            if "mean" in stats:
                                zone_stats["mean"] = float(np.mean(data))
                            if "min" in stats:
                                zone_stats["min"] = float(np.min(data))
                            if "max" in stats:
                                zone_stats["max"] = float(np.max(data))
                            if "sum" in stats:
                                zone_stats["sum"] = float(np.sum(data))
                            if "count" in stats:
                                zone_stats["count"] = len(data)
                            if "std" in stats:
                                zone_stats["std"] = float(np.std(data))
                        else:
                            for stat in stats:
                                zone_stats[stat] = None
                        
                        results.append(zone_stats)
                        
                    except Exception as zone_error:
                        logger.warning(f"Error processing zone {idx}: {zone_error}")
                        results.append({"zone_id": idx, "error": str(zone_error)})
            
            df = pd.DataFrame(results)
            logger.info(f"Calculated zonal stats for {len(df)} zones")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating raster stats: {e}")
            raise ValueError(f"Failed to calculate raster statistics: {e}") from e
    
    def fetch_osm_data(
        self,
        tags: Dict[str, Any],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        place: Optional[str] = None,
        timeout: int = 180
    ) -> "gpd.GeoDataFrame":
        """
        Fetch data from OpenStreetMap using OSMnx.
        
        Args:
            tags: OSM tag filters (e.g., {'building': True, 'natural': 'water'})
            bbox: Bounding box (south, west, north, east) in WGS84
            place: Place name for geocoding (alternative to bbox)
            timeout: Request timeout in seconds
            
        Returns:
            GeoDataFrame with OSM features
        """
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("OSMnx is required for fetching OSM data")
        
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for OSM data operations")
        
        try:
            ox.settings.timeout = timeout
            ox.settings.cache_folder = str(self.cache_dir)
            
            if bbox:
                # Input bbox format: (south, west, north, east) in WGS84
                # OSMnx features_from_bbox expects: (north, south, east, west)
                south, west, north, east = bbox
                gdf = ox.features_from_bbox(
                    bbox=(north, south, east, west),
                    tags=tags
                )
            elif place:
                gdf = ox.features_from_place(place, tags=tags)
            else:
                raise ValueError("Either bbox or place must be provided")
            
            logger.info(f"Fetched {len(gdf)} OSM features with tags {tags}")
            return gdf
            
        except Exception as e:
            logger.error(f"Error fetching OSM data: {e}")
            raise ValueError(f"Failed to fetch OSM data: {e}") from e
    
    @staticmethod
    def spatial_join(
        left: "gpd.GeoDataFrame",
        right: "gpd.GeoDataFrame",
        predicate: str = "intersects",
        how: str = "inner"
    ) -> "gpd.GeoDataFrame":
        """
        Join two GeoDataFrames based on spatial relationship.
        
        Args:
            left: Left GeoDataFrame
            right: Right GeoDataFrame
            predicate: Spatial relationship (intersects, contains, within)
            how: Join type (inner, left, right)
            
        Returns:
            Joined GeoDataFrame
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for spatial join")
        
        try:
            # Ensure CRS match
            if left.crs != right.crs:
                logger.info(f"Transforming right CRS from {right.crs} to {left.crs}")
                right = right.to_crs(left.crs)
            
            result = gpd.sjoin(left, right, predicate=predicate, how=how)
            logger.info(f"Spatial join: {len(result)} features (predicate={predicate})")
            return result
            
        except Exception as e:
            logger.error(f"Error in spatial join: {e}")
            raise ValueError(f"Failed to perform spatial join: {e}") from e
    
    @staticmethod
    def dissolve(
        gdf: "gpd.GeoDataFrame",
        by: Optional[str] = None,
        aggfunc: str = "first"
    ) -> "gpd.GeoDataFrame":
        """
        Dissolve/merge geometries, optionally by a grouping column.
        
        Args:
            gdf: Input GeoDataFrame
            by: Column to group by (None for dissolving all into one)
            aggfunc: Aggregation function for non-geometry columns
            
        Returns:
            Dissolved GeoDataFrame
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for dissolve operation")
        
        try:
            if by:
                result = gdf.dissolve(by=by, aggfunc=aggfunc)
            else:
                # Dissolve all into single geometry
                dissolved_geom = unary_union(gdf.geometry)
                result = gpd.GeoDataFrame(
                    {"geometry": [dissolved_geom]},
                    crs=gdf.crs
                )
            
            logger.info(f"Dissolved to {len(result)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error dissolving geometries: {e}")
            raise ValueError(f"Failed to dissolve geometries: {e}") from e
    
    @staticmethod
    def clip_to_bounds(
        gdf: "gpd.GeoDataFrame",
        bounds: Union[Tuple[float, float, float, float], "gpd.GeoDataFrame"]
    ) -> "gpd.GeoDataFrame":
        """
        Clip a GeoDataFrame to a bounding box or polygon.
        
        Args:
            gdf: Input GeoDataFrame
            bounds: Bounding box (minx, miny, maxx, maxy) or mask GeoDataFrame
            
        Returns:
            Clipped GeoDataFrame
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for clip operation")
        
        try:
            if isinstance(bounds, tuple):
                # Create bounding box geometry
                bbox_geom = box(*bounds)
                mask_gdf = gpd.GeoDataFrame(
                    {"geometry": [bbox_geom]},
                    crs=gdf.crs
                )
            else:
                mask_gdf = bounds
                if mask_gdf.crs != gdf.crs:
                    mask_gdf = mask_gdf.to_crs(gdf.crs)
            
            result = gpd.clip(gdf, mask_gdf)
            logger.info(f"Clipped to {len(result)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error clipping: {e}")
            raise ValueError(f"Failed to clip geometries: {e}") from e
    
    @staticmethod
    def calculate_area(
        gdf: "gpd.GeoDataFrame",
        unit: str = "sqm"
    ) -> "gpd.GeoDataFrame":
        """
        Calculate area of geometries in specified units.
        
        Args:
            gdf: Input GeoDataFrame with polygon geometries
            unit: Output unit (sqm, sqkm, ha, acres)
            
        Returns:
            GeoDataFrame with area column added
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for area calculation")
        
        try:
            result = gdf.copy()
            
            # Project if geographic CRS
            if result.crs and result.crs.is_geographic:
                centroid = result.geometry.unary_union.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                hemisphere = "N" if centroid.y >= 0 else "S"
                epsg = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
                work_gdf = result.to_crs(f"EPSG:{epsg}")
            else:
                work_gdf = result
            
            # Calculate area in square meters
            area_sqm = work_gdf.geometry.area
            
            # Convert to requested unit
            unit_conversions = {
                "sqm": 1.0,
                "sqkm": 1e-6,
                "ha": 1e-4,
                "acres": 2.47105e-4
            }
            
            factor = unit_conversions.get(unit, 1.0)
            result[f"area_{unit}"] = area_sqm * factor
            
            logger.info(f"Calculated area for {len(result)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating area: {e}")
            raise ValueError(f"Failed to calculate area: {e}") from e
    
    @staticmethod
    def intersection(
        gdf1: "gpd.GeoDataFrame",
        gdf2: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """
        Find intersection of two GeoDataFrames.
        
        Args:
            gdf1: First GeoDataFrame
            gdf2: Second GeoDataFrame
            
        Returns:
            GeoDataFrame with intersection geometries
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for intersection")
        
        try:
            # Ensure CRS match
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)
            
            result = gpd.overlay(gdf1, gdf2, how="intersection")
            logger.info(f"Intersection: {len(result)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error in intersection: {e}")
            raise ValueError(f"Failed to calculate intersection: {e}") from e
    
    @staticmethod
    def union(
        gdf1: "gpd.GeoDataFrame",
        gdf2: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """
        Create union of two GeoDataFrames.
        
        Args:
            gdf1: First GeoDataFrame
            gdf2: Second GeoDataFrame
            
        Returns:
            GeoDataFrame with union of all geometries
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for union")
        
        try:
            # Ensure CRS match
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)
            
            result = gpd.overlay(gdf1, gdf2, how="union")
            logger.info(f"Union: {len(result)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error in union: {e}")
            raise ValueError(f"Failed to calculate union: {e}") from e
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for LangChain integration.
        
        Returns:
            List of tool definition dictionaries
        """
        return [
            {
                "name": "load_vector_layer",
                "description": "Load vector geospatial data from GeoJSON, Shapefile, or GeoPackage",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to vector file"},
                        "layer": {"type": "string", "description": "Layer name (optional)"},
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "buffer_geometry",
                "description": "Create buffer zones around geometries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "distance": {"type": "number", "description": "Buffer distance in meters"},
                        "resolution": {"type": "integer", "description": "Buffer resolution"}
                    },
                    "required": ["distance"]
                }
            },
            {
                "name": "calculate_raster_stats",
                "description": "Calculate zonal statistics for raster within vector boundaries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raster_path": {"type": "string", "description": "Path to raster file"},
                        "vector_path": {"type": "string", "description": "Path to vector file"},
                        "stats": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Statistics to calculate"
                        }
                    },
                    "required": ["raster_path", "vector_path"]
                }
            },
            {
                "name": "fetch_osm_data",
                "description": "Fetch geospatial data from OpenStreetMap",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "object", "description": "OSM tags to filter"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Bounding box (south, west, north, east)"
                        },
                        "place": {"type": "string", "description": "Place name"}
                    },
                    "required": ["tags"]
                }
            },
            {
                "name": "spatial_join",
                "description": "Join GeoDataFrames based on spatial relationship",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicate": {
                            "type": "string",
                            "enum": ["intersects", "contains", "within"],
                            "description": "Spatial relationship"
                        },
                        "how": {
                            "type": "string",
                            "enum": ["inner", "left", "right"],
                            "description": "Join type"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "dissolve",
                "description": "Dissolve/merge geometries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "by": {"type": "string", "description": "Column to group by"}
                    },
                    "required": []
                }
            }
        ]
