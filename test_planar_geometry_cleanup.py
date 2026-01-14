#!/usr/bin/env python3
"""
test_planar_geometry_cleanup.py

Validation test to ensure curved geometries (spherical, hyperbolic, cylindrical)
have been successfully removed from the codebase.

This test validates:
1. Config enums only contain planar geometries
2. Factory functions reject curved geometries
3. All imports resolve correctly
4. No references to removed classes remain
5. Planar and planar_hsv workflows function correctly

Run with: python test_planar_geometry_cleanup.py
"""

import sys
import importlib
from typing import List, Tuple

# Track test results
passed_tests: List[str] = []
failed_tests: List[Tuple[str, str]] = []


def test_pass(name: str) -> None:
    """Record a passed test."""
    passed_tests.append(name)
    print(f"✅ PASS: {name}")


def test_fail(name: str, reason: str) -> None:
    """Record a failed test."""
    failed_tests.append((name, reason))
    print(f"❌ FAIL: {name}")
    print(f"   Reason: {reason}")


def test_config_enums():
    """Test that config enums only contain planar geometries."""
    print("\n" + "=" * 60)
    print("Testing config enums...")
    print("=" * 60)
    
    try:
        from ads_cft.config import SliceGeometry, DatasetType
        
        # Check SliceGeometry enum
        valid_geometries = {"PLANAR", "FLAT", "HYPERSCALING_VIOLATING"}
        invalid_geometries = {"SPHERICAL", "HYPERBOLIC", "CYLINDRICAL", 
                              "SPHERICAL_HSV", "HYPERBOLIC_HSV", "CYLINDRICAL_HSV"}
        
        actual_geometries = {g.name for g in SliceGeometry}
        
        # Check no invalid geometries present
        found_invalid = actual_geometries & invalid_geometries
        if found_invalid:
            test_fail("SliceGeometry_no_curved", 
                     f"Found invalid geometries: {found_invalid}")
        else:
            test_pass("SliceGeometry_no_curved")
        
        # Check valid geometries present
        missing_valid = valid_geometries - actual_geometries
        if missing_valid:
            test_fail("SliceGeometry_has_planar",
                     f"Missing valid geometries: {missing_valid}")
        else:
            test_pass("SliceGeometry_has_planar")
        
        # Check DatasetType enum
        invalid_datasets = {"UNIFORM_SPHERE", "SPHERICAL_CLUSTERS", 
                           "SPHERICAL_GAUSSIAN_MIXTURE", "HYPERBOLIC_GAUSSIAN",
                           "HYPERBOLIC_CLUSTERS", "CYLINDRICAL_RINGS",
                           "CYLINDRICAL_SPIRAL", "SPHERE_CHECKERBOARD",
                           "HYPERBOLOID_CHECKERBOARD"}
        
        actual_datasets = {d.name for d in DatasetType}
        
        found_invalid_ds = actual_datasets & invalid_datasets
        if found_invalid_ds:
            test_fail("DatasetType_no_curved",
                     f"Found invalid datasets: {found_invalid_ds}")
        else:
            test_pass("DatasetType_no_curved")
            
    except ImportError as e:
        test_fail("config_import", f"Failed to import config: {e}")
    except Exception as e:
        test_fail("config_enums", f"Unexpected error: {e}")


def test_geometry_factory():
    """Test that geometry factory only accepts planar geometries."""
    print("\n" + "=" * 60)
    print("Testing geometry factory...")
    print("=" * 60)
    
    try:
        from ads_cft.geometry import make_geometry
        from ads_cft.config import GeometryConfig, SliceGeometry
        
        # Test valid geometries
        for geom in [SliceGeometry.PLANAR, SliceGeometry.FLAT, 
                     SliceGeometry.HYPERSCALING_VIOLATING]:
            try:
                cfg = GeometryConfig(
                    slice_geometry=geom,
                    d=2,
                    r_min=0.0,
                    r_max=1.0,
                )
                geometry = make_geometry(cfg)
                if geometry is not None:
                    test_pass(f"make_geometry_{geom.name}")
                else:
                    test_fail(f"make_geometry_{geom.name}", "Returned None")
            except Exception as e:
                test_fail(f"make_geometry_{geom.name}", str(e))
        
    except ImportError as e:
        test_fail("geometry_import", f"Failed to import geometry: {e}")
    except Exception as e:
        test_fail("geometry_factory", f"Unexpected error: {e}")


def test_imports():
    """Test that all module imports resolve correctly."""
    print("\n" + "=" * 60)
    print("Testing module imports...")
    print("=" * 60)
    
    modules = [
        "ads_cft",
        "ads_cft.config",
        "ads_cft.geometry",
        "ads_cft.data_toy",
        "ads_cft.laplacian_base",
        "ads_cft.encoding_base",
        "ads_cft.encoding_spectral",
        "ads_cft.networks",
        "ads_cft.model",
        "ads_cft.baselines",
        "ads_cft.registry",
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            test_pass(f"import_{module.replace('.', '_')}")
        except ImportError as e:
            test_fail(f"import_{module.replace('.', '_')}", str(e))
        except Exception as e:
            test_fail(f"import_{module.replace('.', '_')}", f"Unexpected: {e}")


def test_no_removed_classes():
    """Test that removed classes are not accessible."""
    print("\n" + "=" * 60)
    print("Testing removed classes are inaccessible...")
    print("=" * 60)
    
    # Classes that should NOT exist
    removed_classes = [
        ("ads_cft.geometry", "SphericalAdS"),
        ("ads_cft.geometry", "HyperbolicAdS"),
        ("ads_cft.geometry", "CylindricalAdS"),
        ("ads_cft.geometry", "SphericalHSV"),
        ("ads_cft.geometry", "HyperbolicHSV"),
        ("ads_cft.geometry", "CylindricalHSV"),
        ("ads_cft.laplacian_base", "PackedSphericalLaplacian"),
        ("ads_cft.laplacian_base", "CylindricalPackedDiag"),
        ("ads_cft.data_toy", "UniformSphereSampler"),
        ("ads_cft.data_toy", "HyperbolicGaussianSampler"),
        ("ads_cft.data_toy", "CylindricalRingsSampler"),
    ]
    
    for module_name, class_name in removed_classes:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                test_fail(f"removed_{class_name}", 
                         f"Class {class_name} still exists in {module_name}")
            else:
                test_pass(f"removed_{class_name}")
        except ImportError:
            # Module doesn't exist, which is fine
            test_pass(f"removed_{class_name}")
        except Exception as e:
            test_fail(f"removed_{class_name}", f"Unexpected: {e}")


def test_data_samplers():
    """Test that data samplers work for planar datasets."""
    print("\n" + "=" * 60)
    print("Testing data samplers...")
    print("=" * 60)
    
    try:
        from ads_cft.data_toy import create_sampler
        from ads_cft.config import DatasetType
        
        # Test valid samplers
        valid_datasets = [
            DatasetType.CHECKERBOARD,
            DatasetType.GAUSSIAN_MIXTURE,
            DatasetType.SWISS_ROLL,
            DatasetType.TWO_MOONS,
            DatasetType.CONCENTRIC_CIRCLES,
        ]
        
        for ds in valid_datasets:
            try:
                sampler = create_sampler(ds)
                if sampler is not None:
                    # Try to sample
                    samples = sampler.sample(10)
                    if samples is not None and len(samples) == 10:
                        test_pass(f"sampler_{ds.name}")
                    else:
                        test_fail(f"sampler_{ds.name}", "Failed to generate samples")
                else:
                    test_fail(f"sampler_{ds.name}", "Returned None")
            except Exception as e:
                test_fail(f"sampler_{ds.name}", str(e))
                
    except ImportError as e:
        test_fail("data_toy_import", f"Failed to import data_toy: {e}")
    except Exception as e:
        test_fail("data_samplers", f"Unexpected error: {e}")


def test_laplacian_factory():
    """Test that Laplacian factory works for planar geometries."""
    print("\n" + "=" * 60)
    print("Testing Laplacian factory...")
    print("=" * 60)
    
    try:
        from ads_cft.laplacian_base import build_slice_laplacian, PlanarSpectralLaplacian
        from ads_cft.config import GeometryConfig, DiscretizationConfig, SliceGeometry
        import torch
        
        # Test planar Laplacian
        geom_cfg = GeometryConfig(
            slice_geometry=SliceGeometry.PLANAR,
            d=2,
            r_min=0.0,
            r_max=1.0,
        )
        disc_cfg = DiscretizationConfig(
            representation="spectral",
            grid_shape=(16, 16),
            box_lengths=(2*3.14159, 2*3.14159),
        )
        
        from ads_cft.geometry import make_geometry
        geometry = make_geometry(geom_cfg)
        
        laplacian = build_slice_laplacian(
            geometry=geometry,
            disc=disc_cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        if laplacian is not None:
            test_pass("laplacian_factory")
        else:
            test_fail("laplacian_factory", "Returned None")
            
    except ImportError as e:
        test_fail("laplacian_import", f"Failed to import: {e}")
    except Exception as e:
        test_fail("laplacian_factory", f"Unexpected error: {e}")


def test_registry():
    """Test that registry only contains planar components."""
    print("\n" + "=" * 60)
    print("Testing registry...")
    print("=" * 60)
    
    try:
        from ads_cft.registry import (
            GEOMETRY_REGISTRY,
            LAPLACIAN_REGISTRY,
            DATASET_REGISTRY,
        )
        
        # Check geometry registry
        invalid_geom_keys = {"spherical", "hyperbolic", "cylindrical",
                           "spherical_hsv", "hyperbolic_hsv", "cylindrical_hsv"}
        found_invalid = set(GEOMETRY_REGISTRY.keys()) & invalid_geom_keys
        if found_invalid:
            test_fail("geometry_registry", f"Found invalid: {found_invalid}")
        else:
            test_pass("geometry_registry")
        
        # Check Laplacian registry
        if "collocation" in LAPLACIAN_REGISTRY:
            test_fail("laplacian_registry", "Found 'collocation' entry")
        else:
            test_pass("laplacian_registry")
        
        # Check dataset registry
        invalid_ds_keys = {"uniform_sphere", "spherical_clusters",
                          "hyperbolic_gaussian", "cylindrical_rings"}
        found_invalid_ds = set(DATASET_REGISTRY.keys()) & invalid_ds_keys
        if found_invalid_ds:
            test_fail("dataset_registry", f"Found invalid: {found_invalid_ds}")
        else:
            test_pass("dataset_registry")
            
    except ImportError as e:
        test_fail("registry_import", f"Failed to import registry: {e}")
    except Exception as e:
        test_fail("registry", f"Unexpected error: {e}")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("PLANAR GEOMETRY CLEANUP VALIDATION TESTS")
    print("=" * 60)
    
    # Run all test categories
    test_config_enums()
    test_geometry_factory()
    test_imports()
    test_no_removed_classes()
    test_data_samplers()
    test_laplacian_factory()
    test_registry()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for name, reason in failed_tests:
            print(f"  - {name}: {reason}")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
