"""Stimulus module — product catalogs, register specs, prompt construction,
LLM generation, and quality-gate functions.

This module defines:
    - REAL_PRODUCTS: 40 real products (8 categories x 5)
    - FICTIONAL_PRODUCTS: 40 fictional products (8 categories x 5)
    - REGISTER_SPECS: 5 linguistic registers
    - CROSS_GENERATOR_SUBSET_IDS: 10 products for cross-generator comparison
    - build_generation_prompt(): construct a generation prompt
    - generate_all_stimuli(): batch generation via Claude + GPT-4
    - run_bow_baseline(): TF-IDF + logistic regression baseline
    - check_register_distinctiveness(): inter-/intra-register distance check
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Real Product Catalog — 8 categories x 5 products
# ---------------------------------------------------------------------------

REAL_PRODUCTS: list[dict[str, Any]] = [
    # ── Oral Care ──────────────────────────────────────────────────────────
    {
        "id": "oral_care_001",
        "name": "Colgate Total",
        "category": "oral_care",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "stannous fluoride 0.454%",
            "fluoride_ppm": 1450,
            "tube_size_oz": 4.8,
            "whitening_claim": True,
        },
        "distinguishing_features": [
            "12-hour antibacterial protection",
            "whole mouth health",
        ],
    },
    {
        "id": "oral_care_002",
        "name": "Crest Pro-Health Advanced",
        "category": "oral_care",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "stannous fluoride 0.454%",
            "fluoride_ppm": 1100,
            "tube_size_oz": 5.1,
            "sensitivity_relief": True,
            "enamel_shield": True,
        },
        "distinguishing_features": [
            "active foam technology",
            "gum and enamel defense",
        ],
    },
    {
        "id": "oral_care_003",
        "name": "Sensodyne Pronamel",
        "category": "oral_care",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "potassium nitrate 5%",
            "fluoride_ppm": 1450,
            "tube_size_oz": 4.0,
            "ph_level": 7.0,
        },
        "distinguishing_features": [
            "acid erosion protection",
            "reharden enamel micro-layer",
        ],
    },
    {
        "id": "oral_care_004",
        "name": "Tom's of Maine Whole Care",
        "category": "oral_care",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "sodium monofluorophosphate 0.76%",
            "fluoride_ppm": 1000,
            "tube_size_oz": 4.0,
            "natural_certified": True,
        },
        "distinguishing_features": [
            "no artificial preservatives or flavors",
            "naturally derived silica whitening",
        ],
    },
    {
        "id": "oral_care_005",
        "name": "Oral-B Gum & Enamel Repair",
        "category": "oral_care",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "stannous fluoride 0.454%",
            "fluoride_ppm": 1350,
            "tube_size_oz": 3.4,
            "gum_repair_claim": True,
            "remineralization_index": 3.2,
        },
        "distinguishing_features": [
            "stabilized stannous fluoride complex",
            "reverses early gum damage in 4 weeks",
        ],
    },
    # ── Pet Food ───────────────────────────────────────────────────────────
    {
        "id": "pet_food_001",
        "name": "Purina Pro Plan Adult Chicken",
        "category": "pet_food",
        "is_fictional": False,
        "core_attributes": {
            "protein_pct": 30,
            "fat_pct": 18,
            "fiber_pct": 3.0,
            "bag_weight_lb": 35,
            "kcal_per_cup": 476,
        },
        "distinguishing_features": [
            "live probiotics for digestive health",
            "real chicken as first ingredient",
        ],
    },
    {
        "id": "pet_food_002",
        "name": "Blue Buffalo Life Protection",
        "category": "pet_food",
        "is_fictional": False,
        "core_attributes": {
            "protein_pct": 26,
            "fat_pct": 15,
            "fiber_pct": 5.0,
            "bag_weight_lb": 30,
            "kcal_per_cup": 378,
        },
        "distinguishing_features": [
            "LifeSource Bits antioxidant blend",
            "no poultry by-product meals",
        ],
    },
    {
        "id": "pet_food_003",
        "name": "Hill's Science Diet Adult",
        "category": "pet_food",
        "is_fictional": False,
        "core_attributes": {
            "protein_pct": 24,
            "fat_pct": 16.5,
            "fiber_pct": 2.5,
            "bag_weight_lb": 30,
            "kcal_per_cup": 363,
        },
        "distinguishing_features": [
            "veterinarian recommended No. 1 brand",
            "balanced mineral blend for heart and kidney health",
        ],
    },
    {
        "id": "pet_food_004",
        "name": "Royal Canin Medium Adult",
        "category": "pet_food",
        "is_fictional": False,
        "core_attributes": {
            "protein_pct": 25,
            "fat_pct": 14,
            "fiber_pct": 3.3,
            "bag_weight_lb": 30,
            "kcal_per_cup": 339,
        },
        "distinguishing_features": [
            "breed-size-specific kibble shape",
            "optimal skin and coat support with EPA/DHA",
        ],
    },
    {
        "id": "pet_food_005",
        "name": "Orijen Original Grain-Free",
        "category": "pet_food",
        "is_fictional": False,
        "core_attributes": {
            "protein_pct": 38,
            "fat_pct": 18,
            "fiber_pct": 5.0,
            "bag_weight_lb": 25,
            "kcal_per_cup": 449,
        },
        "distinguishing_features": [
            "WholePrey ratio with organs and bone",
            "two-thirds fresh or raw animal ingredients",
        ],
    },
    # ── Home Cleaning ─────────────────────────────────────────────────────
    {
        "id": "home_cleaning_001",
        "name": "Method All-Purpose Cleaner",
        "category": "home_cleaning",
        "is_fictional": False,
        "core_attributes": {
            "active_surfactant": "decyl glucoside",
            "volume_fl_oz": 28,
            "ph_level": 8.5,
            "biodegradable": True,
        },
        "distinguishing_features": [
            "plant-based formula",
            "cradle-to-cradle certified",
        ],
    },
    {
        "id": "home_cleaning_002",
        "name": "Clorox Disinfecting Wipes",
        "category": "home_cleaning",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "alkyl dimethyl benzyl ammonium chloride 0.184%",
            "wipe_count": 75,
            "kill_rate_pct": 99.9,
            "contact_time_sec": 30,
        },
        "distinguishing_features": [
            "kills SARS-CoV-2 virus on hard surfaces",
            "bleach-free and EPA registered",
        ],
    },
    {
        "id": "home_cleaning_003",
        "name": "Mrs. Meyer's Multi-Surface Everyday Cleaner",
        "category": "home_cleaning",
        "is_fictional": False,
        "core_attributes": {
            "active_surfactant": "sodium lauryl sulfate",
            "volume_fl_oz": 16,
            "ph_level": 7.2,
            "essential_oil_blend": "lavender",
        },
        "distinguishing_features": [
            "garden-inspired scent line",
            "derived from plant and mineral ingredients",
        ],
    },
    {
        "id": "home_cleaning_004",
        "name": "Lysol Power Bathroom Cleaner",
        "category": "home_cleaning",
        "is_fictional": False,
        "core_attributes": {
            "active_ingredient": "hydrochloric acid 9.5%",
            "volume_fl_oz": 32,
            "kill_rate_pct": 99.9,
            "contact_time_sec": 10,
            "ph_level": 1.5,
        },
        "distinguishing_features": [
            "10x dissolving power on soap scum",
            "thick gel clings to vertical surfaces",
        ],
    },
    {
        "id": "home_cleaning_005",
        "name": "Seventh Generation Dish Liquid",
        "category": "home_cleaning",
        "is_fictional": False,
        "core_attributes": {
            "active_surfactant": "sodium coco-sulfate",
            "volume_fl_oz": 19,
            "ph_level": 7.0,
            "biodegradable": True,
            "usda_biobased_pct": 95,
        },
        "distinguishing_features": [
            "USDA Certified Biobased 95%",
            "hypoallergenic fragrance-free formula",
        ],
    },
    # ── Sports Nutrition ──────────────────────────────────────────────────
    {
        "id": "sports_nutrition_001",
        "name": "Optimum Nutrition Gold Standard Whey",
        "category": "sports_nutrition",
        "is_fictional": False,
        "core_attributes": {
            "protein_per_serving_g": 24,
            "calories_per_serving": 120,
            "bcaa_per_serving_g": 5.5,
            "serving_size_g": 30.4,
            "container_servings": 74,
        },
        "distinguishing_features": [
            "whey protein isolate as primary source",
            "Informed Sport certified batch-tested",
        ],
    },
    {
        "id": "sports_nutrition_002",
        "name": "Gatorade Thirst Quencher",
        "category": "sports_nutrition",
        "is_fictional": False,
        "core_attributes": {
            "calories_per_serving": 80,
            "sodium_mg": 160,
            "potassium_mg": 45,
            "sugar_g": 21,
            "volume_fl_oz": 20,
        },
        "distinguishing_features": [
            "electrolyte replacement for athletic performance",
            "scientifically proven formula since 1965",
        ],
    },
    {
        "id": "sports_nutrition_003",
        "name": "Clif Bar Original Energy Bar",
        "category": "sports_nutrition",
        "is_fictional": False,
        "core_attributes": {
            "protein_per_serving_g": 10,
            "calories_per_serving": 250,
            "fiber_g": 4,
            "bar_weight_g": 68,
            "sugar_g": 21,
        },
        "distinguishing_features": [
            "70% organic ingredients",
            "sustained energy blend of rolled oats and brown rice syrup",
        ],
    },
    {
        "id": "sports_nutrition_004",
        "name": "Nuun Sport Electrolyte Tablets",
        "category": "sports_nutrition",
        "is_fictional": False,
        "core_attributes": {
            "calories_per_serving": 15,
            "sodium_mg": 300,
            "potassium_mg": 150,
            "magnesium_mg": 25,
            "tablets_per_tube": 10,
        },
        "distinguishing_features": [
            "effervescent tablet dissolves in water",
            "low sugar with plant-based sweetener",
        ],
    },
    {
        "id": "sports_nutrition_005",
        "name": "BSN Syntha-6 Protein Powder",
        "category": "sports_nutrition",
        "is_fictional": False,
        "core_attributes": {
            "protein_per_serving_g": 22,
            "calories_per_serving": 200,
            "fat_per_serving_g": 6,
            "serving_size_g": 47,
            "container_servings": 48,
        },
        "distinguishing_features": [
            "six-protein blend with micellar casein",
            "ultra-premium taste profile with MCTs and fiber",
        ],
    },
    # ── Baby Care ─────────────────────────────────────────────────────────
    {
        "id": "baby_care_001",
        "name": "Pampers Swaddlers Size 3",
        "category": "baby_care",
        "is_fictional": False,
        "core_attributes": {
            "weight_range_lb": "16-28",
            "diaper_count": 168,
            "absorbency_layers": 3,
            "wetness_indicator": True,
        },
        "distinguishing_features": [
            "Absorb Away liner pulls wetness from skin",
            "umbilical cord notch for newborn sizes",
        ],
    },
    {
        "id": "baby_care_002",
        "name": "Enfamil NeuroPro Infant Formula",
        "category": "baby_care",
        "is_fictional": False,
        "core_attributes": {
            "dha_mg_per_100kcal": 17,
            "calories_per_fl_oz": 20,
            "container_oz": 20.7,
            "mfgm_enriched": True,
            "iron_mg_per_100kcal": 1.8,
        },
        "distinguishing_features": [
            "MFGM and DHA for brain support",
            "closest to breast milk in fat-protein globule structure",
        ],
    },
    {
        "id": "baby_care_003",
        "name": "Aquaphor Baby Healing Ointment",
        "category": "baby_care",
        "is_fictional": False,
        "core_attributes": {
            "petrolatum_pct": 41,
            "tube_size_oz": 3.0,
            "fragrance_free": True,
            "preservative_free": True,
        },
        "distinguishing_features": [
            "multi-purpose for diaper rash and dry skin",
            "pediatrician recommended barrier ointment",
        ],
    },
    {
        "id": "baby_care_004",
        "name": "Babyganics Shampoo & Body Wash",
        "category": "baby_care",
        "is_fictional": False,
        "core_attributes": {
            "volume_fl_oz": 16,
            "ph_level": 5.5,
            "tear_free": True,
            "plant_derived_pct": 98,
        },
        "distinguishing_features": [
            "NeoNourish seed oil blend",
            "non-allergenic and dermatologist tested",
        ],
    },
    {
        "id": "baby_care_005",
        "name": "Huggies Natural Care Wipes",
        "category": "baby_care",
        "is_fictional": False,
        "core_attributes": {
            "wipe_count": 528,
            "water_content_pct": 99,
            "thickness_mm": 0.6,
            "fragrance_free": True,
        },
        "distinguishing_features": [
            "99% purified water and plant-based materials",
            "triple clean layers for mess-free changes",
        ],
    },
    # ── Coffee/Beverage ───────────────────────────────────────────────────
    {
        "id": "coffee_beverage_001",
        "name": "Starbucks Pike Place Roast K-Cup",
        "category": "coffee_beverage",
        "is_fictional": False,
        "core_attributes": {
            "roast_level": "medium",
            "caffeine_mg": 130,
            "pod_count": 72,
            "brew_volume_oz": 8,
        },
        "distinguishing_features": [
            "smooth balanced flavor with cocoa and toasted nut notes",
            "100% arabica ethically sourced coffee",
        ],
    },
    {
        "id": "coffee_beverage_002",
        "name": "Celsius Sparkling Energy Drink",
        "category": "coffee_beverage",
        "is_fictional": False,
        "core_attributes": {
            "caffeine_mg": 200,
            "calories_per_can": 10,
            "can_volume_fl_oz": 12,
            "sugar_g": 0,
            "green_tea_extract_mg": 270,
        },
        "distinguishing_features": [
            "MetaPlus proprietary thermogenic blend",
            "clinically shown to boost metabolism",
        ],
    },
    {
        "id": "coffee_beverage_003",
        "name": "Nespresso Vertuo Medium Roast",
        "category": "coffee_beverage",
        "is_fictional": False,
        "core_attributes": {
            "roast_level": "medium",
            "caffeine_mg": 170,
            "pod_count": 30,
            "brew_volume_oz": 7.77,
            "intensity_scale_1_13": 6,
        },
        "distinguishing_features": [
            "centrifusion barcode-scanned brewing technology",
            "crema layer from high-pressure extraction",
        ],
    },
    {
        "id": "coffee_beverage_004",
        "name": "Liquid Death Mountain Water",
        "category": "coffee_beverage",
        "is_fictional": False,
        "core_attributes": {
            "volume_fl_oz": 19.2,
            "tds_ppm": 130,
            "ph_level": 8.2,
            "packaging": "infinitely recyclable aluminum",
        },
        "distinguishing_features": [
            "sourced from Austrian Alps limestone aquifer",
            "tallboy can format targeting non-alcohol occasions",
        ],
    },
    {
        "id": "coffee_beverage_005",
        "name": "Califia Farms Oat Barista Blend",
        "category": "coffee_beverage",
        "is_fictional": False,
        "core_attributes": {
            "volume_fl_oz": 32,
            "calories_per_serving": 120,
            "protein_g_per_serving": 3,
            "fat_g_per_serving": 7,
            "shelf_stable": False,
        },
        "distinguishing_features": [
            "microfoam performance rivaling dairy",
            "whole-grain North American oats with no gums",
        ],
    },
    # ── Skincare ──────────────────────────────────────────────────────────
    {
        "id": "skincare_001",
        "name": "CeraVe Moisturizing Cream",
        "category": "skincare",
        "is_fictional": False,
        "core_attributes": {
            "ceramide_complex": "1, 3, 6-II",
            "jar_size_oz": 19,
            "hyaluronic_acid": True,
            "fragrance_free": True,
        },
        "distinguishing_features": [
            "MVE controlled-release technology over 24 hours",
            "developed with dermatologists",
        ],
    },
    {
        "id": "skincare_002",
        "name": "La Roche-Posay Anthelios Melt-in Milk SPF 100",
        "category": "skincare",
        "is_fictional": False,
        "core_attributes": {
            "spf_rating": 100,
            "uva_pf": 46,
            "volume_fl_oz": 3.0,
            "water_resistant_min": 80,
        },
        "distinguishing_features": [
            "Cell-Ox Shield antioxidant technology",
            "La Roche-Posay thermal spring water",
        ],
    },
    {
        "id": "skincare_003",
        "name": "The Ordinary Niacinamide 10% + Zinc 1%",
        "category": "skincare",
        "is_fictional": False,
        "core_attributes": {
            "niacinamide_pct": 10.0,
            "zinc_pca_pct": 1.0,
            "volume_ml": 30,
            "ph_range": "5.0-6.5",
        },
        "distinguishing_features": [
            "high-strength vitamin and mineral blemish formula",
            "water-based lightweight serum texture",
        ],
    },
    {
        "id": "skincare_004",
        "name": "Neutrogena Hydro Boost Water Gel",
        "category": "skincare",
        "is_fictional": False,
        "core_attributes": {
            "hyaluronic_acid_mw_kda": 50,
            "jar_size_oz": 1.7,
            "oil_free": True,
            "absorption_time_sec": 15,
        },
        "distinguishing_features": [
            "hyaluronic acid gel matrix locks in hydration",
            "non-comedogenic oil-free formula",
        ],
    },
    {
        "id": "skincare_005",
        "name": "EltaMD UV Clear Broad-Spectrum SPF 46",
        "category": "skincare",
        "is_fictional": False,
        "core_attributes": {
            "spf_rating": 46,
            "zinc_oxide_pct": 9.0,
            "niacinamide_pct": 5.0,
            "volume_fl_oz": 1.7,
        },
        "distinguishing_features": [
            "lightly tinted transparent zinc for acne-prone skin",
            "Skin Cancer Foundation recommended daily use",
        ],
    },
    # ── Smart Home ────────────────────────────────────────────────────────
    {
        "id": "smart_home_001",
        "name": "Philips Hue White and Color Ambiance A19",
        "category": "smart_home",
        "is_fictional": False,
        "core_attributes": {
            "lumens": 1100,
            "wattage": 9.5,
            "color_temperature_range_k": "2000-6500",
            "color_gamut": "16 million",
            "lifespan_hours": 25000,
        },
        "distinguishing_features": [
            "Zigbee 3.0 with Matter compatibility",
            "syncs with Spotify and gaming via Hue Entertainment",
        ],
    },
    {
        "id": "smart_home_002",
        "name": "Ring Video Doorbell 4",
        "category": "smart_home",
        "is_fictional": False,
        "core_attributes": {
            "resolution_p": 1080,
            "field_of_view_deg": 160,
            "night_vision": True,
            "battery_life_months": 6,
        },
        "distinguishing_features": [
            "Pre-Roll Video captures 4 seconds before motion",
            "two-way talk with noise cancellation",
        ],
    },
    {
        "id": "smart_home_003",
        "name": "Ecobee Smart Thermostat Premium",
        "category": "smart_home",
        "is_fictional": False,
        "core_attributes": {
            "temperature_accuracy_f": 0.5,
            "display_size_in": 3.5,
            "smart_sensor_range_ft": 60,
            "energy_savings_pct": 26,
        },
        "distinguishing_features": [
            "built-in air quality monitor and Alexa speaker",
            "eco+ community energy savings program",
        ],
    },
    {
        "id": "smart_home_004",
        "name": "Aqara Presence Sensor FP2",
        "category": "smart_home",
        "is_fictional": False,
        "core_attributes": {
            "detection_range_m": 5,
            "detection_zones": 30,
            "refresh_rate_hz": 5,
            "power_consumption_w": 2.0,
        },
        "distinguishing_features": [
            "mmWave radar detects static human presence",
            "zone-based multi-person tracking up to 5 targets",
        ],
    },
    {
        "id": "smart_home_005",
        "name": "TP-Link Kasa Smart Plug Mini EP10",
        "category": "smart_home",
        "is_fictional": False,
        "core_attributes": {
            "max_load_amps": 15,
            "wifi_band": "2.4 GHz",
            "energy_monitoring": False,
            "size_depth_in": 2.6,
        },
        "distinguishing_features": [
            "compact design does not block adjacent outlets",
            "away mode randomly toggles to simulate occupancy",
        ],
    },
]

# ---------------------------------------------------------------------------
# Fictional Product Catalog — 8 categories x 5 products
# ---------------------------------------------------------------------------

FICTIONAL_PRODUCTS: list[dict[str, Any]] = [
    # ── Oral Care (fictional) ─────────────────────────────────────────────
    {
        "id": "oral_care_f001",
        "name": "AeroMint ProShield Toothpaste",
        "category": "oral_care",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "hydroxyapatite 15%",
            "fluoride_ppm": 0,
            "tube_size_oz": 5.0,
            "remineralization_index": 4.8,
        },
        "distinguishing_features": [
            "nano-hydroxyapatite fluoride-free enamel repair",
            "microencapsulated mint release over 6 hours",
        ],
    },
    {
        "id": "oral_care_f002",
        "name": "BrightCore Enzyme Gel",
        "category": "oral_care",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "lactoperoxidase enzyme complex 2.5%",
            "fluoride_ppm": 850,
            "tube_size_oz": 3.5,
            "biofilm_disruption_score": 8.7,
        },
        "distinguishing_features": [
            "enzyme-based biofilm dissolution",
            "probiotic delivery system for oral microbiome",
        ],
    },
    {
        "id": "oral_care_f003",
        "name": "VelvetSmile Charcoal Elixir",
        "category": "oral_care",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "activated bamboo charcoal 8%",
            "fluoride_ppm": 500,
            "tube_size_oz": 4.2,
            "stain_removal_index": 9.1,
        },
        "distinguishing_features": [
            "dual-phase charcoal-silica whitening system",
            "pH-buffered at 6.8 for gentle daily use",
        ],
    },
    {
        "id": "oral_care_f004",
        "name": "NovaDent Ion Shield Rinse",
        "category": "oral_care",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "silver-copper ion complex 0.02%",
            "fluoride_ppm": 225,
            "volume_fl_oz": 16,
            "antimicrobial_duration_hr": 18,
        },
        "distinguishing_features": [
            "metal-ion antimicrobial mouthwash",
            "zero alcohol with dual-ion freshness lock",
        ],
    },
    {
        "id": "oral_care_f005",
        "name": "PureSeal Ceramic Coat Paste",
        "category": "oral_care",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "bioceramic microsphere complex 12%",
            "fluoride_ppm": 1200,
            "tube_size_oz": 3.8,
            "coating_duration_hr": 10,
        },
        "distinguishing_features": [
            "ceramic microsphere tooth-coating technology",
            "forms protective glaze after brushing",
        ],
    },
    # ── Pet Food (fictional) ──────────────────────────────────────────────
    {
        "id": "pet_food_f001",
        "name": "TerraHound Ancestral Blend",
        "category": "pet_food",
        "is_fictional": True,
        "core_attributes": {
            "protein_pct": 42,
            "fat_pct": 20,
            "fiber_pct": 4.5,
            "bag_weight_lb": 28,
            "kcal_per_cup": 510,
        },
        "distinguishing_features": [
            "freeze-dried raw coated kibble",
            "insect-protein enriched for sustainability",
        ],
    },
    {
        "id": "pet_food_f002",
        "name": "NourishPaws Calm & Digest",
        "category": "pet_food",
        "is_fictional": True,
        "core_attributes": {
            "protein_pct": 28,
            "fat_pct": 12,
            "fiber_pct": 7.0,
            "bag_weight_lb": 22,
            "kcal_per_cup": 330,
        },
        "distinguishing_features": [
            "L-theanine calming complex 200mg/cup",
            "prebiotic-fermented pumpkin base",
        ],
    },
    {
        "id": "pet_food_f003",
        "name": "OceanTail Salmon & Kelp Formula",
        "category": "pet_food",
        "is_fictional": True,
        "core_attributes": {
            "protein_pct": 34,
            "fat_pct": 16,
            "fiber_pct": 3.8,
            "bag_weight_lb": 20,
            "kcal_per_cup": 420,
        },
        "distinguishing_features": [
            "wild-caught Pacific salmon with kelp fiber",
            "omega-3 DHA at 0.5% for coat shine",
        ],
    },
    {
        "id": "pet_food_f004",
        "name": "VerdePet Plant-Forward Kibble",
        "category": "pet_food",
        "is_fictional": True,
        "core_attributes": {
            "protein_pct": 25,
            "fat_pct": 10,
            "fiber_pct": 8.5,
            "bag_weight_lb": 18,
            "kcal_per_cup": 290,
        },
        "distinguishing_features": [
            "50% plant-based protein from chickpea and lentil",
            "carbon-neutral production certified",
        ],
    },
    {
        "id": "pet_food_f005",
        "name": "PrimeCoat Joint & Vitality Formula",
        "category": "pet_food",
        "is_fictional": True,
        "core_attributes": {
            "protein_pct": 30,
            "fat_pct": 15,
            "fiber_pct": 4.0,
            "bag_weight_lb": 32,
            "kcal_per_cup": 385,
        },
        "distinguishing_features": [
            "glucosamine HCl 1500mg/kg for joint support",
            "turmeric-curcumin anti-inflammatory blend",
        ],
    },
    # ── Home Cleaning (fictional) ─────────────────────────────────────────
    {
        "id": "home_cleaning_f001",
        "name": "ZymoClear Enzyme All-Surface Spray",
        "category": "home_cleaning",
        "is_fictional": True,
        "core_attributes": {
            "active_surfactant": "protease-amylase enzyme blend",
            "volume_fl_oz": 24,
            "ph_level": 7.5,
            "biodegradable": True,
            "enzyme_concentration_ppm": 800,
        },
        "distinguishing_features": [
            "triple-enzyme formula dissolves organic residue",
            "refill pod system reduces plastic waste by 85%",
        ],
    },
    {
        "id": "home_cleaning_f002",
        "name": "NanoGuard Antimicrobial Surface Seal",
        "category": "home_cleaning",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "photocatalytic TiO2 nanocoating",
            "volume_fl_oz": 16,
            "protection_duration_days": 30,
            "kill_rate_pct": 99.97,
        },
        "distinguishing_features": [
            "light-activated titanium dioxide self-cleaning surface",
            "single application lasts 30 days",
        ],
    },
    {
        "id": "home_cleaning_f003",
        "name": "PureStream Ozone Disinfection Tabs",
        "category": "home_cleaning",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "sodium percarbonate + ozone activator",
            "tab_count": 60,
            "kill_rate_pct": 99.99,
            "contact_time_sec": 45,
            "ph_level": 10.5,
        },
        "distinguishing_features": [
            "dissolves in water to produce ozone micro-bubbles",
            "chemical-free residue after activation",
        ],
    },
    {
        "id": "home_cleaning_f004",
        "name": "TerraSoft Probiotic Floor Wash",
        "category": "home_cleaning",
        "is_fictional": True,
        "core_attributes": {
            "active_surfactant": "Bacillus spore probiotic blend",
            "volume_fl_oz": 32,
            "ph_level": 6.5,
            "probiotic_cfu_per_ml": 1000000,
        },
        "distinguishing_features": [
            "live probiotic cleaning continues for 72 hours",
            "safe for sealed hardwood and natural stone",
        ],
    },
    {
        "id": "home_cleaning_f005",
        "name": "AirLoom Textile Refresh Mist",
        "category": "home_cleaning",
        "is_fictional": True,
        "core_attributes": {
            "active_ingredient": "cyclodextrin odor-capture complex",
            "volume_fl_oz": 10,
            "ph_level": 6.8,
            "drying_time_min": 5,
        },
        "distinguishing_features": [
            "cyclodextrin molecules trap and neutralize odors",
            "safe for upholstery, curtains, and delicate fabrics",
        ],
    },
    # ── Sports Nutrition (fictional) ──────────────────────────────────────
    {
        "id": "sports_nutrition_f001",
        "name": "VeloFuel Endurance Gel",
        "category": "sports_nutrition",
        "is_fictional": True,
        "core_attributes": {
            "calories_per_serving": 110,
            "carb_blend_ratio": "2:1 maltodextrin:fructose",
            "sodium_mg": 200,
            "caffeine_mg": 40,
            "packet_weight_g": 35,
        },
        "distinguishing_features": [
            "dual-carb transport system for 90g/hr absorption",
            "isotonic consistency requires no water chaser",
        ],
    },
    {
        "id": "sports_nutrition_f002",
        "name": "IronForge Creatine HMB Complex",
        "category": "sports_nutrition",
        "is_fictional": True,
        "core_attributes": {
            "creatine_monohydrate_g": 5,
            "hmb_g": 3,
            "calories_per_serving": 20,
            "serving_size_g": 12,
            "container_servings": 60,
        },
        "distinguishing_features": [
            "combined creatine and HMB for lean mass gains",
            "micronized for instant mixing without grit",
        ],
    },
    {
        "id": "sports_nutrition_f003",
        "name": "AquaPulse Hydration Powder",
        "category": "sports_nutrition",
        "is_fictional": True,
        "core_attributes": {
            "sodium_mg": 500,
            "potassium_mg": 200,
            "magnesium_mg": 60,
            "calories_per_serving": 5,
            "servings_per_pouch": 40,
        },
        "distinguishing_features": [
            "high-sodium oral rehydration formula",
            "trace mineral complex from Great Salt Lake source",
        ],
    },
    {
        "id": "sports_nutrition_f004",
        "name": "BioRecovr Post-Workout Blend",
        "category": "sports_nutrition",
        "is_fictional": True,
        "core_attributes": {
            "protein_per_serving_g": 30,
            "tart_cherry_extract_mg": 500,
            "calories_per_serving": 180,
            "leucine_g": 4,
            "serving_size_g": 50,
        },
        "distinguishing_features": [
            "tart cherry extract for inflammation reduction",
            "4:1 carb-to-protein ratio for glycogen replenishment",
        ],
    },
    {
        "id": "sports_nutrition_f005",
        "name": "NitroEdge Pre-Workout Surge",
        "category": "sports_nutrition",
        "is_fictional": True,
        "core_attributes": {
            "caffeine_mg": 300,
            "beta_alanine_g": 3.2,
            "citrulline_malate_g": 8,
            "calories_per_serving": 10,
            "serving_size_g": 18,
        },
        "distinguishing_features": [
            "clinical-dose citrulline for nitric oxide pump",
            "sustained-release caffeine from green tea and guarana",
        ],
    },
    # ── Baby Care (fictional) ─────────────────────────────────────────────
    {
        "id": "baby_care_f001",
        "name": "CloudNest Adaptive-Fit Diaper",
        "category": "baby_care",
        "is_fictional": True,
        "core_attributes": {
            "weight_range_lb": "12-24",
            "diaper_count": 200,
            "absorbency_layers": 5,
            "wetness_indicator": True,
            "compostable_shell": True,
        },
        "distinguishing_features": [
            "five-layer bamboo-fiber core with SAP beads",
            "compostable outer shell certified TUV OK Compost",
        ],
    },
    {
        "id": "baby_care_f002",
        "name": "TinyBloom Probiotic Baby Lotion",
        "category": "baby_care",
        "is_fictional": True,
        "core_attributes": {
            "volume_fl_oz": 8,
            "probiotic_lysate_pct": 2.0,
            "ph_level": 5.2,
            "fragrance_free": True,
        },
        "distinguishing_features": [
            "heat-treated probiotic lysate supports skin barrier",
            "ceramide NP complex for eczema-prone infant skin",
        ],
    },
    {
        "id": "baby_care_f003",
        "name": "PureCradle Organic Formula Stage 1",
        "category": "baby_care",
        "is_fictional": True,
        "core_attributes": {
            "dha_mg_per_100kcal": 22,
            "calories_per_fl_oz": 20,
            "container_oz": 24.0,
            "organic_certified": True,
            "iron_mg_per_100kcal": 2.0,
        },
        "distinguishing_features": [
            "EU-organic whole-milk base with A2 beta-casein",
            "DHA from algal oil, no fish-derived ingredients",
        ],
    },
    {
        "id": "baby_care_f004",
        "name": "SnugWrap Merino Sleep Sack",
        "category": "baby_care",
        "is_fictional": True,
        "core_attributes": {
            "tog_rating": 2.5,
            "weight_range_lb": "14-26",
            "merino_wool_gsm": 180,
            "oeko_tex_certified": True,
        },
        "distinguishing_features": [
            "merino wool temperature regulation 64-77°F comfort zone",
            "two-way zipper for easy nighttime changes",
        ],
    },
    {
        "id": "baby_care_f005",
        "name": "LittleGuard Mineral Sunscreen Stick SPF 50",
        "category": "baby_care",
        "is_fictional": True,
        "core_attributes": {
            "spf_rating": 50,
            "zinc_oxide_pct": 22,
            "stick_weight_oz": 0.7,
            "water_resistant_min": 80,
            "reef_safe": True,
        },
        "distinguishing_features": [
            "non-nano zinc oxide safe for 6+ months",
            "twist-up stick for mess-free face application",
        ],
    },
    # ── Coffee/Beverage (fictional) ───────────────────────────────────────
    {
        "id": "coffee_beverage_f001",
        "name": "BrewCraft Cold Nitro Concentrate",
        "category": "coffee_beverage",
        "is_fictional": True,
        "core_attributes": {
            "caffeine_mg_per_fl_oz": 35,
            "volume_fl_oz": 16,
            "roast_level": "dark",
            "nitrogen_infused": True,
            "concentrate_ratio": "1:3",
        },
        "distinguishing_features": [
            "nitrogen-infused shelf-stable cold brew concentrate",
            "24-hour slow steep with single-origin Colombian beans",
        ],
    },
    {
        "id": "coffee_beverage_f002",
        "name": "MorningRise Mushroom Latte Mix",
        "category": "coffee_beverage",
        "is_fictional": True,
        "core_attributes": {
            "caffeine_mg": 50,
            "lion_mane_extract_mg": 500,
            "calories_per_serving": 35,
            "packets_per_box": 30,
            "sugar_g": 1,
        },
        "distinguishing_features": [
            "lion's mane and chaga dual-extract nootropic blend",
            "oat milk powder included for creamy one-step prep",
        ],
    },
    {
        "id": "coffee_beverage_f003",
        "name": "Solara Sparkling Yerba Mate",
        "category": "coffee_beverage",
        "is_fictional": True,
        "core_attributes": {
            "caffeine_mg": 150,
            "calories_per_can": 20,
            "can_volume_fl_oz": 12,
            "sugar_g": 4,
            "antioxidant_orac": 12000,
        },
        "distinguishing_features": [
            "organic yerba mate with added polyphenol complex",
            "light carbonation with passion fruit finish",
        ],
    },
    {
        "id": "coffee_beverage_f004",
        "name": "ArcticBrew Protein Iced Coffee",
        "category": "coffee_beverage",
        "is_fictional": True,
        "core_attributes": {
            "caffeine_mg": 180,
            "protein_g_per_serving": 20,
            "calories_per_serving": 150,
            "volume_fl_oz": 15,
            "sugar_g": 3,
        },
        "distinguishing_features": [
            "whey isolate protein blended with cold brew",
            "lactose-free with added MCT oil for sustained energy",
        ],
    },
    {
        "id": "coffee_beverage_f005",
        "name": "ZenLeaf Ceremonial Matcha Sachets",
        "category": "coffee_beverage",
        "is_fictional": True,
        "core_attributes": {
            "caffeine_mg": 70,
            "l_theanine_mg": 40,
            "sachet_count": 20,
            "grade": "ceremonial A",
            "umami_index": 8.5,
        },
        "distinguishing_features": [
            "stone-ground Uji first-flush tencha leaves",
            "single-serve nitrogen-flushed sachets for freshness",
        ],
    },
    # ── Skincare (fictional) ──────────────────────────────────────────────
    {
        "id": "skincare_f001",
        "name": "LuminVeil Adaptive Barrier Cream",
        "category": "skincare",
        "is_fictional": True,
        "core_attributes": {
            "ceramide_types": "1, 3, 6-II, EOS",
            "jar_size_oz": 2.0,
            "squalane_pct": 8.0,
            "ph_level": 5.5,
        },
        "distinguishing_features": [
            "quad-ceramide complex with plant squalane",
            "microbiome-friendly preservative system",
        ],
    },
    {
        "id": "skincare_f002",
        "name": "GlassGlow Peptide-C Serum",
        "category": "skincare",
        "is_fictional": True,
        "core_attributes": {
            "vitamin_c_pct": 15,
            "copper_peptide_ppm": 200,
            "volume_ml": 30,
            "ph_level": 3.2,
        },
        "distinguishing_features": [
            "stabilized L-ascorbic acid with GHK-Cu peptide",
            "airless pump preserves potency for 90 days",
        ],
    },
    {
        "id": "skincare_f003",
        "name": "AquaShell UV Defense Fluid SPF 55",
        "category": "skincare",
        "is_fictional": True,
        "core_attributes": {
            "spf_rating": 55,
            "uva_pf": 30,
            "volume_ml": 50,
            "water_resistant_min": 40,
            "finish": "invisible matte",
        },
        "distinguishing_features": [
            "hybrid organic-mineral filter system",
            "weightless fluid texture with blue-light protection",
        ],
    },
    {
        "id": "skincare_f004",
        "name": "DewDrop Centella Recovery Gel",
        "category": "skincare",
        "is_fictional": True,
        "core_attributes": {
            "centella_extract_pct": 70,
            "madecassoside_mg_per_ml": 5,
            "volume_ml": 50,
            "ph_level": 5.8,
        },
        "distinguishing_features": [
            "high-concentration centella gel for irritated skin",
            "cooling hydrogel base with panthenol 3%",
        ],
    },
    {
        "id": "skincare_f005",
        "name": "NightForge Retinal Sleep Mask",
        "category": "skincare",
        "is_fictional": True,
        "core_attributes": {
            "retinal_pct": 0.1,
            "bakuchiol_pct": 1.0,
            "jar_size_oz": 1.5,
            "encapsulation": "liposomal",
        },
        "distinguishing_features": [
            "liposomal retinaldehyde with bakuchiol buffering",
            "overnight mask texture forms breathable film",
        ],
    },
    # ── Smart Home (fictional) ────────────────────────────────────────────
    {
        "id": "smart_home_f001",
        "name": "LumenArc Adaptive Ceiling Panel",
        "category": "smart_home",
        "is_fictional": True,
        "core_attributes": {
            "lumens": 4000,
            "wattage": 36,
            "color_temperature_range_k": "1800-8000",
            "panel_size_in": "24x24",
            "lifespan_hours": 50000,
        },
        "distinguishing_features": [
            "circadian rhythm auto-adjustment from dawn to dusk",
            "edge-lit micro-LED panel with zero flicker",
        ],
    },
    {
        "id": "smart_home_f002",
        "name": "SentryView 360 Outdoor Camera",
        "category": "smart_home",
        "is_fictional": True,
        "core_attributes": {
            "resolution_p": 2160,
            "field_of_view_deg": 360,
            "night_vision_range_ft": 100,
            "local_storage_gb": 256,
            "ip_rating": "IP67",
        },
        "distinguishing_features": [
            "on-device AI person/vehicle/animal classification",
            "dual-band Wi-Fi 6E with local NVR fallback storage",
        ],
    },
    {
        "id": "smart_home_f003",
        "name": "ClimaSense Whole-Home Air Monitor",
        "category": "smart_home",
        "is_fictional": True,
        "core_attributes": {
            "pm25_accuracy_pct": 95,
            "co2_range_ppm": "400-5000",
            "voc_sensor": True,
            "sensor_lifespan_years": 10,
            "display_size_in": 4.0,
        },
        "distinguishing_features": [
            "seven-sensor array: PM2.5, CO2, VOC, temp, humidity, noise, light",
            "HVAC integration triggers ventilation at thresholds",
        ],
    },
    {
        "id": "smart_home_f004",
        "name": "FlowTap Smart Water Valve",
        "category": "smart_home",
        "is_fictional": True,
        "core_attributes": {
            "pipe_diameter_in": 1.0,
            "flow_rate_gpm": 15,
            "leak_detection_sensitivity_ml": 50,
            "battery_backup_hr": 72,
        },
        "distinguishing_features": [
            "ultrasonic flow measurement with auto-shutoff on leak",
            "72-hour battery backup maintains protection during outages",
        ],
    },
    {
        "id": "smart_home_f005",
        "name": "EcoGrid Smart Power Strip",
        "category": "smart_home",
        "is_fictional": True,
        "core_attributes": {
            "outlet_count": 6,
            "usb_c_ports": 2,
            "max_load_watts": 1800,
            "energy_monitoring": True,
            "surge_protection_joules": 4000,
        },
        "distinguishing_features": [
            "per-outlet energy monitoring with vampire-draw auto-cutoff",
            "surge protection at 4000 joules with LED status per outlet",
        ],
    },
]

# ---------------------------------------------------------------------------
# Register Specifications
# ---------------------------------------------------------------------------

REGISTER_SPECS: dict[str, dict[str, str]] = {
    "marketing": {
        "voice": (
            "Second-person direct address ('you', 'your'). Brand-confident, "
            "aspirational, benefit-forward."
        ),
        "tone": (
            "Enthusiastic and persuasive; emphasizes transformation and lifestyle "
            "improvement. Uses power words (revolutionary, unleash, discover)."
        ),
        "structure": (
            "Short punchy sentences or fragments. Headline + 2-3 benefit bullets or "
            "a flowing paragraph that builds desire. Ends with call-to-action or "
            "tagline."
        ),
        "vocabulary": (
            "Superlatives (best-in-class, ultimate), sensory adjectives (silky, "
            "crisp, bold), emotional triggers (confidence, freedom, peace of mind). "
            "Avoids jargon unless it sounds impressive."
        ),
        "example_source": "Product landing pages, Amazon A+ content, DTC brand copy.",
    },
    "regulatory": {
        "voice": (
            "Third-person impersonal. Formal, precise, authoritative. "
            "Uses passive constructions and nominal style."
        ),
        "tone": (
            "Neutral, objective, clinical. No emotional language. Prioritizes "
            "accuracy and completeness over readability."
        ),
        "structure": (
            "Dense paragraph or numbered sections. Starts with product "
            "identification, active ingredients and concentrations, indications, "
            "directions for use, warnings. Follows regulatory labeling conventions."
        ),
        "vocabulary": (
            "Technical nomenclature (INCI names, USP grades), quantitative "
            "specifications, regulatory terms (indicated for, contraindicated, "
            "adverse reactions). Units are SI or USP-standard."
        ),
        "example_source": (
            "FDA drug facts panels, EU safety data sheets, AAFCO pet food "
            "guaranteed analysis labels."
        ),
    },
    "casual_social": {
        "voice": (
            "First-person singular ('I', 'my'). Conversational, authentic, "
            "relatable peer voice."
        ),
        "tone": (
            "Genuine enthusiasm or honest critique. Uses humor, hyperbole, "
            "and colloquialisms. Emotionally transparent."
        ),
        "structure": (
            "Informal flowing paragraph like a social media caption or product "
            "review. May start mid-thought. Uses ellipses, dashes, exclamations. "
            "No formal heading or bullet points."
        ),
        "vocabulary": (
            "Everyday language, slang (game-changer, lowkey, obsessed, vibe), "
            "contractions (it's, don't, I'm), filler words (honestly, literally, "
            "basically). Brand names used casually."
        ),
        "example_source": (
            "Instagram captions, Reddit product reviews, TikTok voiceover scripts, "
            "casual blog posts."
        ),
    },
    "patent": {
        "voice": (
            "Third-person impersonal. Highly formal, legalistic, exhaustively "
            "precise. Inventor-assignee framing."
        ),
        "tone": (
            "Detached, methodical, claims-oriented. Every statement narrows scope "
            "or establishes novelty. No subjective evaluation."
        ),
        "structure": (
            "Opens with field of invention, then detailed description referencing "
            "compositions, methods, and embodiments. Uses 'comprising', "
            "'wherein', 'the method of claim 1'. Numbered elements."
        ),
        "vocabulary": (
            "Legal-technical: 'embodiment', 'thereof', 'plurality', 'substantially', "
            "'configured to'. Chemical formulas, process parameters, and ranges "
            "(e.g., 'between about 5% and about 15% by weight')."
        ),
        "example_source": (
            "USPTO patent applications, EPO published specifications, "
            "WIPO PCT claims."
        ),
    },
    "journalistic": {
        "voice": (
            "Third-person omniscient. Balanced reporter's voice with attributed "
            "quotes and sourced claims."
        ),
        "tone": (
            "Informative, measured, lightly analytical. May include mild skepticism "
            "or context about market trends. Avoids both hype and hostility."
        ),
        "structure": (
            "Inverted-pyramid: lead sentence with newsworthy angle, context "
            "paragraph, supporting details, expert or consumer quote, and "
            "closing perspective. 3-5 sentences."
        ),
        "vocabulary": (
            "Journalistic conventions: attribution verbs (claims, according to, "
            "notes), hedging (appears to, is expected to), market/industry terms "
            "(market share, consumer segment, product category). Accessible but "
            "precise."
        ),
        "example_source": (
            "Consumer Reports reviews, Wirecutter picks, New York Times product "
            "coverage, trade journal articles."
        ),
    },
}

# ---------------------------------------------------------------------------
# Cross-Generator Subset — 5 real + 5 fictional across 5 categories
# ---------------------------------------------------------------------------

CROSS_GENERATOR_SUBSET_IDS: list[str] = [
    # Real products (5 categories)
    "oral_care_001",          # Colgate Total
    "pet_food_003",           # Hill's Science Diet
    "sports_nutrition_001",   # ON Gold Standard Whey
    "skincare_001",           # CeraVe Moisturizing Cream
    "smart_home_003",         # Ecobee Smart Thermostat
    # Fictional products (5 categories)
    "home_cleaning_f001",     # ZymoClear Enzyme Spray
    "baby_care_f003",         # PureCradle Organic Formula
    "coffee_beverage_f002",   # MorningRise Mushroom Latte
    "sports_nutrition_f004",  # BioRecovr Post-Workout
    "skincare_f002",          # GlassGlow Peptide-C Serum
]

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_ALL_PRODUCTS: list[dict[str, Any]] = REAL_PRODUCTS + FICTIONAL_PRODUCTS

_PRODUCT_BY_ID: dict[str, dict[str, Any]] = {p["id"]: p for p in _ALL_PRODUCTS}


def get_product(product_id: str) -> dict[str, Any]:
    """Return a product dict by its id, or raise KeyError."""
    return _PRODUCT_BY_ID[product_id]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_generation_prompt(
    product: dict[str, Any],
    register: str,
    variant_index: int,
) -> str:
    """Build a prompt that instructs an LLM to generate a product description.

    The prompt includes the product's core attributes and distinguishing
    features, plus the full register specification (voice, tone, structure,
    vocabulary, example_source).  Variant 0 and variant 1 receive different
    diversity instructions to encourage lexical variation across the pair.

    Args:
        product: A product dict with keys id, name, category, is_fictional,
            core_attributes, and distinguishing_features.
        register: One of the five register keys (marketing, regulatory,
            casual_social, patent, journalistic).
        variant_index: 0 or 1; controls the diversity instruction.

    Returns:
        A string prompt ready to send to a chat-completion API.
    """
    spec = REGISTER_SPECS[register]

    # Format core attributes for the prompt
    attr_lines = []
    for key, value in product["core_attributes"].items():
        attr_lines.append(f"  - {key}: {value}")
    attr_block = "\n".join(attr_lines)

    features_block = "\n".join(
        f"  - {f}" for f in product["distinguishing_features"]
    )

    # Diversity instruction differs by variant
    if variant_index == 0:
        diversity = (
            "Write in a straightforward style for this register. "
            "Prioritize clarity and natural phrasing."
        )
    else:
        diversity = (
            "Write a DIFFERENT version from your usual approach for this register. "
            "Vary your sentence structure, word choice, and opening strategy. "
            "Still respect the register constraints but explore alternative "
            "phrasings and a distinct rhetorical angle."
        )

    prompt = f"""\
Write a product description for the following item in the **{register.replace('_', ' ')}** register.

## Product Information
- **Name**: {product["name"]}
- **Category**: {product["category"].replace('_', ' ').title()}
- **Core Attributes**:
{attr_block}
- **Distinguishing Features**:
{features_block}

## Register Specification
- **Voice**: {spec["voice"]}
- **Tone**: {spec["tone"]}
- **Structure**: {spec["structure"]}
- **Vocabulary**: {spec["vocabulary"]}
- **Example sources**: {spec["example_source"]}

## Constraints
1. Target length: 80-150 words (hard limits: 50-200 words).
2. ALL core attributes (numerical values, percentages, specific ingredients) must be \
conveyed in the text — do not omit any.
3. Do NOT use the product name as a heading or title; weave it naturally into the text.
4. Do NOT include meta-commentary (e.g., "Here is a description…").
5. Output ONLY the product description text.

## Diversity Instruction
{diversity}
"""
    return prompt


# ---------------------------------------------------------------------------
# Token counting utility
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting.

    This is a simple heuristic; for more precise counts, use tiktoken.
    """
    return len(text.split())


# ---------------------------------------------------------------------------
# Core attribute coverage validation
# ---------------------------------------------------------------------------


def _check_attribute_coverage(
    text: str,
    core_attributes: dict[str, Any],
) -> tuple[float, list[str]]:
    """Check what fraction of core attributes are mentioned in text.

    Uses keyword extraction and fuzzy matching:
    - Numeric values: checks if the number appears in text
    - Boolean True: checks if the attribute name (snake_case -> words) appears
    - String values: checks if key words from the value appear in text

    Returns:
        (coverage_ratio, list_of_missing_attribute_keys)
    """
    text_lower = text.lower()
    missing = []

    for key, value in core_attributes.items():
        found = False

        if isinstance(value, bool):
            if value:
                # Check for the attribute concept in text
                words = key.replace("_", " ").lower().split()
                if any(w in text_lower for w in words if len(w) > 3):
                    found = True
        elif isinstance(value, (int, float)):
            # Check if the number appears in text
            str_val = str(value)
            if str_val in text_lower:
                found = True
            # Also try without trailing .0
            if isinstance(value, float) and value == int(value):
                if str(int(value)) in text_lower:
                    found = True
        elif isinstance(value, str):
            # Check if significant words from the value appear
            value_words = re.findall(r"[a-z]{3,}", value.lower())
            if value_words:
                matches = sum(1 for w in value_words if w in text_lower)
                if matches / len(value_words) >= 0.3:
                    found = True

        if not found:
            missing.append(key)

    total = len(core_attributes)
    covered = total - len(missing)
    return (covered / total if total > 0 else 1.0, missing)


# ---------------------------------------------------------------------------
# Stimulus generation
# ---------------------------------------------------------------------------


def generate_all_stimuli(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all stimulus texts via Claude (primary) and GPT-4 (cross-generator).

    Generates:
      - Primary set: 80 products x 5 registers x 2 variants = 800 via Claude
      - Cross-generator set: 10-product subset x 5 registers x 2 variants = 100 via GPT-4

    Saves intermediate results to data/stimuli.json every 20 stimuli for
    crash recovery.  On restart, existing stimulus_ids are skipped.

    Args:
        config: The CONFIG dict from run.py with keys like output_dir,
            registers, variants_per_product, token_range_target,
            token_range_accept, token_range_hard_reject, cross_generator_subset_size.

    Returns:
        List of stimulus dicts (see module docstring for schema).
    """
    import anthropic
    import openai

    output_dir = Path(config.get("output_dir", "data/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stimuli_path = output_dir / "stimuli.json"

    # Load existing stimuli for resume support
    existing_stimuli: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if stimuli_path.exists():
        with open(stimuli_path, "r") as f:
            existing_stimuli = json.load(f)
        existing_ids = {s["stimulus_id"] for s in existing_stimuli}
        print(f"[stimuli] Loaded {len(existing_stimuli)} existing stimuli, resuming.")

    all_stimuli = list(existing_stimuli)

    registers = config.get(
        "registers",
        ["marketing", "regulatory", "casual_social", "patent", "journalistic"],
    )
    n_variants = config.get("variants_per_product", 2)
    target_low, target_high = config.get("token_range_target", (80, 150))
    accept_low, accept_high = config.get("token_range_accept", (50, 200))
    hard_low, hard_high = config.get("token_range_hard_reject", (40, 250))
    max_retries = 3

    # ── Build generation queue ────────────────────────────────────────────
    queue: list[tuple[dict[str, Any], str, int, str]] = []  # (product, register, variant, generator)

    # Primary: all products via Claude
    for product in _ALL_PRODUCTS:
        for reg in registers:
            for v in range(n_variants):
                sid = f"{product['id']}_{reg}_v{v}"
                if sid not in existing_ids:
                    queue.append((product, reg, v, "claude"))

    # Cross-generator: subset via GPT-4
    for pid in CROSS_GENERATOR_SUBSET_IDS:
        product = _PRODUCT_BY_ID[pid]
        for reg in registers:
            for v in range(n_variants):
                sid = f"{product['id']}_{reg}_v{v}_gpt4"
                if sid not in existing_ids:
                    queue.append((product, reg, v, "gpt4"))

    if not queue:
        print("[stimuli] All stimuli already generated.")
        return all_stimuli

    print(f"[stimuli] Generating {len(queue)} stimuli ...")

    # ── Initialize API clients ────────────────────────────────────────────
    claude_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    openai_client = openai.OpenAI()        # reads OPENAI_API_KEY

    batch_count = 0

    for product, reg, variant, generator in queue:
        prompt = build_generation_prompt(product, reg, variant)

        if generator == "gpt4":
            sid = f"{product['id']}_{reg}_v{variant}_gpt4"
        else:
            sid = f"{product['id']}_{reg}_v{variant}"

        text = None
        for attempt in range(max_retries):
            try:
                if generator == "claude":
                    response = claude_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=300,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip()
                else:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=300,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.choices[0].message.content.strip()

                # Validate token count
                tc = _count_tokens(text)
                if tc < hard_low or tc > hard_high:
                    print(
                        f"  [retry {attempt+1}] {sid}: {tc} tokens "
                        f"outside hard limits [{hard_low}, {hard_high}]"
                    )
                    text = None
                    continue

                if tc < accept_low or tc > accept_high:
                    print(
                        f"  [warn] {sid}: {tc} tokens outside accept range "
                        f"[{accept_low}, {accept_high}] but within hard limits"
                    )

                break  # success

            except Exception as e:
                print(f"  [error] {sid} attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)  # exponential backoff

        if text is None:
            print(f"  [FAIL] {sid}: exhausted {max_retries} retries, skipping")
            continue

        tc = _count_tokens(text)
        coverage, missing = _check_attribute_coverage(
            text, product["core_attributes"]
        )

        if coverage < 1.0:
            print(
                f"  [warn] {sid}: attribute coverage {coverage:.0%}, "
                f"missing: {missing}"
            )

        stimulus = {
            "stimulus_id": sid,
            "product_id": product["id"],
            "category": product["category"],
            "register": reg,
            "variant": variant,
            "is_fictional": product["is_fictional"],
            "text": text,
            "token_count": tc,
            "generator": generator,
            "core_attributes": product["core_attributes"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        all_stimuli.append(stimulus)
        existing_ids.add(sid)
        batch_count += 1

        # Save every 20 stimuli
        if batch_count % 20 == 0:
            with open(stimuli_path, "w") as f:
                json.dump(all_stimuli, f, indent=2, default=str)
            print(f"  [save] {len(all_stimuli)} stimuli saved to {stimuli_path}")

    # Final save
    with open(stimuli_path, "w") as f:
        json.dump(all_stimuli, f, indent=2, default=str)
    print(f"[stimuli] Generation complete: {len(all_stimuli)} stimuli saved to {stimuli_path}")

    return all_stimuli


# ---------------------------------------------------------------------------
# BoW Baseline (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------


def run_bow_baseline(
    stimuli_path: str | Path = "data/stimuli.json",
) -> dict[str, float]:
    """Train bag-of-words classifiers on three tasks and return accuracies.

    Tasks:
        1. **product** — 40-class product identification (real products only)
        2. **category** — 8-class category classification
        3. **register** — 5-class register classification

    Uses TF-IDF features with logistic regression and 5-fold stratified CV.
    If BoW 40-class product accuracy exceeds 50%, surface features may be
    driving results and stimuli should be reviewed.

    Args:
        stimuli_path: Path to the stimuli JSON file.

    Returns:
        Dict mapping task name to mean cross-validated accuracy.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    with open(stimuli_path, "r") as f:
        stimuli = json.load(f)

    # Filter to Claude-generated stimuli only (primary set)
    primary = [s for s in stimuli if s["generator"] == "claude"]

    if not primary:
        print("[bow] No primary (Claude) stimuli found.")
        return {}

    texts = [s["text"] for s in primary]
    product_labels = [s["product_id"] for s in primary]
    category_labels = [s["category"] for s in primary]
    register_labels = [s["register"] for s in primary]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for task_name, labels in [
        ("product", product_labels),
        ("category", category_labels),
        ("register", register_labels),
    ]:
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
        mean_acc = float(scores.mean())
        results[task_name] = mean_acc
        print(f"[bow] {task_name}: mean accuracy = {mean_acc:.3f} (std = {scores.std():.3f})")

    # Quality gate warning
    if results.get("product", 0) > 0.50:
        print(
            "[bow] WARNING: Product classification accuracy > 50%. "
            "Surface features may be driving results — consider revising stimuli."
        )

    return results


# ---------------------------------------------------------------------------
# Register Distinctiveness Check
# ---------------------------------------------------------------------------


def check_register_distinctiveness(
    stimuli_path: str | Path = "data/stimuli.json",
    threshold_ratio: float = 1.5,
) -> dict[str, Any]:
    """Check that registers are linguistically distinct via TF-IDF cosine distance.

    Computes mean pairwise cosine distance between registers (inter-register)
    and within registers (intra-register) for each product.  If the ratio
    of inter-register to intra-register distance is below *threshold_ratio*,
    a warning is printed.

    Args:
        stimuli_path: Path to the stimuli JSON file.
        threshold_ratio: Minimum ratio of mean inter-register distance to
            mean intra-register distance (default 1.5).

    Returns:
        Dict with keys 'mean_inter', 'mean_intra', 'ratio', 'pass'.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances

    with open(stimuli_path, "r") as f:
        stimuli = json.load(f)

    primary = [s for s in stimuli if s["generator"] == "claude"]

    if not primary:
        print("[register] No primary stimuli found.")
        return {"mean_inter": 0.0, "mean_intra": 0.0, "ratio": 0.0, "pass": False}

    texts = [s["text"] for s in primary]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    dist_matrix = cosine_distances(X)

    # Compute inter-register and intra-register distances
    inter_dists = []
    intra_dists = []

    for i in range(len(primary)):
        for j in range(i + 1, len(primary)):
            d = dist_matrix[i, j]
            same_product = primary[i]["product_id"] == primary[j]["product_id"]
            same_register = primary[i]["register"] == primary[j]["register"]

            if same_product:
                if same_register:
                    # Intra-register: same product, same register, different variant
                    intra_dists.append(d)
                else:
                    # Inter-register: same product, different register
                    inter_dists.append(d)

    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    ratio = mean_inter / mean_intra if mean_intra > 0 else float("inf")
    passed = ratio >= threshold_ratio

    print(f"[register] Mean inter-register distance: {mean_inter:.4f}")
    print(f"[register] Mean intra-register distance: {mean_intra:.4f}")
    print(f"[register] Ratio (inter/intra): {ratio:.2f} (threshold: {threshold_ratio})")

    if not passed:
        print(
            f"[register] WARNING: Ratio {ratio:.2f} < {threshold_ratio}. "
            "Registers may not be sufficiently distinct — consider revising prompts."
        )
    else:
        print("[register] PASS: Registers are sufficiently distinct.")

    return {
        "mean_inter": mean_inter,
        "mean_intra": mean_intra,
        "ratio": ratio,
        "pass": passed,
    }
