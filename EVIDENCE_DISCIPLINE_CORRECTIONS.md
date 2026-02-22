# Evidence-Disciplined Corrections Report

## Corrected JSON Output

```json
{
  "request_id": "0141c6b4ba24",
  "timestamp": "2026-02-04T09:27:19.563914+00:00",
  "processing_time_ms": 15800.45,
  "media_type": "image",
  "authenticity": {
    "score": 44.9,
    "confidence_interval": {
      "lower": 40.0,
      "upper": 47.0,
      "level": 0.95
    },
    "interpretation": "Weak evidence of synthetic origin - manual review recommended"
  },
  "source_identification": {
    "likely_source": null,
    "confidence": null,
    "alternatives": [],
    "is_human_created": null
  },
  "modification_analysis": {
    "likelihood": null,
    "detected": [],
    "compression_detected": null,
    "resize_detected": null
  },
  "provenance": {
    "has_provenance": false,
    "status": "unverified",
    "chain_length": 0,
    "signed_links": 0,
    "original_hash": null
  },
  "ensemble": {
    "models_used": [
      "statistical_baseline",
      "neural_network_detector"
    ],
    "agreement": null,
    "agreement_status": "not_computable",
    "per_model_scores": null
  },
  "out_of_distribution": {
    "is_ood": null,
    "score": null
  },
  "feature_attributions": "not_available",
  "warnings": []
}
```

## Change Log

| Field | Old Value | New Value | Reason |
|-------|-----------|-----------|--------|
| `authenticity.interpretation` | `"Moderate evidence of synthetic generation"` | `"Weak evidence of synthetic origin - manual review recommended"` | FIX A: CI width = 7, ci_upper = 47. Conservative rules require ci_upper <= 30 for "moderate". Downgraded to "weak". |
| `source_identification.confidence` | `0.0` | `null` | FIX B: 0.0 forbidden unless disproving with certainty. Absence of evidence = null. |
| `ensemble.agreement` | `93.0` | `null` | FIX C: Agreement requires per_model_scores. Without actual scores, undefined math is forbidden. |
| `ensemble.agreement_status` | (not present) | `"not_computable"` | FIX C: Explicit state when per_model_scores unavailable. |
| `ensemble.per_model_scores` | `{"statistical_baseline": null, "neural_network_detector": null}` | `null` | FIX C: Null scores cannot justify agreement value. |

## Validation Checklist

| Check | Result | Notes |
|-------|--------|-------|
| No 0.0 implying certainty | ✅ PASS | Source confidence now null |
| No 50.0 implying certainty | ✅ PASS | No placeholder values |
| No false implying certainty | ✅ PASS | All modification flags null |
| All nulls represent uncertainty | ✅ PASS | Every null = "not computed" |
| Interpretation matches evidence | ✅ PASS | Downgraded to "weak" per CI width/position |
| Agreement requires math | ✅ PASS | Null + explicit "not_computable" status |
| Conservative language | ✅ PASS | No escalation without justification |
| No logical inconsistency | ✅ PASS | All fields semantically aligned |

## Files Modified

1. `src/api/detection_api.py`: Updated `get_interpretation()` with conservative thresholds and CI width checks
2. `src/modeling/ood_detector.py`: `GeneratorClassifier.classify()` returns `(None, None)` instead of `(None, 0.0)`
3. `src/api/detection_api.py`: `APIResponse.to_dict()` validates agreement against per_model_scores

## Enforcement Rule Added

**CONTINUOUS FUTURE RULE**: System will now detect and auto-correct:
- Overconfident language not supported by CI
- Certainty claims without evidence
- Undefined mathematical operations
- 0.0/50.0 placeholders
