# Monitoring & Observability Design

> **Mode**: SAFETY MONITORING  
> **Purpose**: Detect silent failures, misuse, and drift  
> **NOT**: Business analytics or growth metrics

---

## 1. METRICS LIST

### 1.1 Service Health Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `request_count_total` | Counter | Total requests received | count |
| `request_count_by_endpoint` | Counter | Requests per endpoint | count |
| `error_count_4xx` | Counter | Client errors | count |
| `error_count_5xx` | Counter | Server errors | count |
| `error_rate_4xx` | Gauge | 4xx rate (rolling 5min) | percent |
| `error_rate_5xx` | Gauge | 5xx rate (rolling 5min) | percent |
| `latency_p50_ms` | Gauge | Median latency | ms |
| `latency_p95_ms` | Gauge | 95th percentile latency | ms |
| `latency_p99_ms` | Gauge | 99th percentile latency | ms |
| `timeout_count` | Counter | Requests that timed out | count |
| `model_loaded` | Gauge | Model load status (1/0) | boolean |
| `uptime_seconds` | Gauge | Service uptime | seconds |

---

### 1.2 Input Characteristics Metrics (NO CONTENT)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `input_resolution_bucket` | Histogram | Image resolution distribution | bucket: small/medium/large/xlarge |
| `input_filesize_bucket` | Histogram | File size distribution | bucket: <1MB/1-5MB/5-10MB |
| `input_format_count` | Counter | Uploads by format | format: jpeg/png/webp |
| `upload_rate_by_ip_hash` | Gauge | Requests per IP (rolling 1min) | ip_hash |

**Resolution Buckets:**
- `small`: < 256px
- `medium`: 256px – 512px
- `large`: 512px – 1024px
- `xlarge`: > 1024px

**File Size Buckets:**
- `tiny`: < 100KB
- `small`: 100KB – 1MB
- `medium`: 1MB – 5MB
- `large`: 5MB – 10MB

---

### 1.3 Output Distribution Metrics (CRITICAL)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `probability_histogram` | Histogram | Synthetic probability distribution | bucket: 0-10/10-20/.../90-100 |
| `probability_bucket_count` | Counter | Results by probability range | bucket: low/borderline/high |
| `probability_mean_rolling` | Gauge | Mean probability (rolling 1hr) | - |
| `probability_stddev_rolling` | Gauge | Std dev (rolling 1hr) | - |
| `confidence_interval_width` | Histogram | CI width distribution | bucket: narrow/medium/wide |
| `borderline_rate` | Gauge | % results in 30-70% range | percent |
| `inconclusive_count` | Counter | "INCONCLUSIVE" interpretations | count |

**Probability Buckets:**
- `low`: 0% – 30%
- `borderline`: 30% – 70%
- `high`: 70% – 100%

**Confidence Interval Width Buckets:**
- `narrow`: < 0.10
- `medium`: 0.10 – 0.20
- `wide`: > 0.20

---

### 1.4 Semantic Integrity Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `interpretation_count` | Counter | Count by interpretation label |
| `missing_version_metadata` | Counter | Responses missing version info |
| `semantic_version_mismatch` | Counter | Version mismatches detected |
| `model_version_in_response` | Gauge | Model version currently serving |

---

### 1.5 Misuse Signal Metrics (COARSE)

| Metric | Type | Description |
|--------|------|-------------|
| `high_rate_ip_count` | Gauge | IPs exceeding rate threshold |
| `repeated_upload_pattern` | Counter | Near-identical upload sequences |
| `automated_pattern_detected` | Counter | Suspected automation signals |
| `burst_traffic_detected` | Counter | Sudden traffic spikes |

---

## 2. LOGGING SCHEMA (JSON)

### 2.1 Request Log Entry

```json
{
  "timestamp": "2026-02-05T01:45:00.000Z",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "endpoint": "/analyze/image",
  "method": "POST",
  "status_code": 200,
  "latency_ms": 245,
  "model_version": "1.0.0",
  "semantic_version": "1.0.0",
  "api_version": "1.0.0",
  
  "input": {
    "resolution_bucket": "large",
    "filesize_bucket": "medium",
    "format": "jpeg"
  },
  
  "output": {
    "probability_bucket": "high",
    "probability_decile": 7,
    "confidence_width_bucket": "narrow",
    "interpretation": "MODERATE-HIGH synthetic likelihood"
  },
  
  "client": {
    "ip_hash": "sha256:abc123...",
    "rate_1min": 3
  },
  
  "flags": {
    "borderline": false,
    "high_rate_client": false,
    "timeout": false
  }
}
```

### 2.2 Error Log Entry

```json
{
  "timestamp": "2026-02-05T01:45:00.000Z",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "endpoint": "/analyze/image",
  "method": "POST",
  "status_code": 500,
  "latency_ms": 5000,
  "error_type": "FEATURE_EXTRACTION_ERROR",
  "error_message": "Could not extract features from image",
  "model_version": "1.0.0",
  
  "input": {
    "resolution_bucket": "small",
    "filesize_bucket": "tiny",
    "format": "jpeg"
  },
  
  "client": {
    "ip_hash": "sha256:abc123..."
  }
}
```

### 2.3 Health Check Log Entry

```json
{
  "timestamp": "2026-02-05T01:45:00.000Z",
  "endpoint": "/health",
  "status_code": 200,
  "service_status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.0,
  "model_version": "1.0.0"
}
```

### 2.4 Fields NOT Logged

| Field | Reason |
|-------|--------|
| Image pixels | Privacy, storage |
| Image content | Privacy |
| Filename | PII risk |
| EXIF data | PII risk |
| User identity | Privacy |
| Raw IP address | Privacy (use hash) |
| Exact probability | Use bucket instead |

---

## 3. ALERT DEFINITIONS

### 3.1 Service Health Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `HighErrorRate5xx` | error_rate_5xx > 5% for 5min | CRITICAL | Page on-call |
| `HighErrorRate4xx` | error_rate_4xx > 20% for 5min | WARNING | Notify team |
| `HighLatencyP95` | latency_p95_ms > 2000 for 5min | WARNING | Investigate |
| `HighLatencyP99` | latency_p99_ms > 5000 for 5min | CRITICAL | Page on-call |
| `TimeoutSpike` | timeout_count > 10 in 5min | WARNING | Investigate |
| `ModelNotLoaded` | model_loaded == 0 | CRITICAL | Page immediately |
| `ServiceDown` | no requests for 5min (during expected hours) | CRITICAL | Page on-call |

---

### 3.2 Output Distribution Alerts (DRIFT)

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `ProbabilityDistributionShift` | mean changes > 2σ from baseline | WARNING | Investigate |
| `BorderlineRateSpike` | borderline_rate > 60% for 1hr | WARNING | Review inputs |
| `BorderlineRateCollapse` | borderline_rate < 10% for 1hr | WARNING | Check model |
| `HighProbabilitySpike` | high_bucket_rate > 80% for 1hr | WARNING | Investigate |
| `LowProbabilitySpike` | low_bucket_rate > 80% for 1hr | WARNING | Investigate |
| `ConfidenceIntervalWidening` | wide_ci_rate > 30% for 1hr | WARNING | Check inputs |

**Baseline Calculation:**
- Rolling 7-day mean/stddev
- Recalculated daily
- Excludes anomalous periods

---

### 3.3 Semantic Integrity Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `MissingVersionMetadata` | missing_version_metadata > 0 | CRITICAL | Immediate fix |
| `SemanticVersionMismatch` | semantic_version != expected | CRITICAL | Rollback check |
| `InconclusiveRateAnomaly` | inconclusive_rate changes > 3σ | WARNING | Investigate |

---

### 3.4 Misuse Pattern Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `HighRateClient` | any IP > 100 req/min for 5min | WARNING | Review |
| `BurstTraffic` | request_count > 10x baseline in 1min | WARNING | Review |
| `RepeatedUploadPattern` | same resolution+size pattern > 50x from IP | INFO | Log for review |
| `AutomationSuspected` | regular interval requests from IP | INFO | Log for review |

**Note:** These are signals only. No automatic blocking. Human review required.

---

### 3.5 Alert Escalation

| Severity | Response Time | Notification |
|----------|---------------|--------------|
| CRITICAL | Immediate | Page on-call + Slack |
| WARNING | 30 minutes | Slack channel |
| INFO | Next business day | Dashboard only |

---

## 4. DASHBOARD LAYOUT

### 4.1 Overview Panel (Top Row)

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  REQUEST RATE   │   ERROR RATE    │    LATENCY      │  MODEL STATUS   │
│                 │                 │                 │                 │
│    1,234/min    │   0.5% (5xx)    │   P50: 120ms    │   ✅ LOADED     │
│    ▲ +12%       │   2.1% (4xx)    │   P95: 450ms    │   v1.0.0        │
│                 │                 │   P99: 890ms    │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

### 4.2 Probability Distribution Panel (Second Row)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROBABILITY DISTRIBUTION (Last 24 Hours)                               │
│                                                                         │
│  100% ┤                                                                 │
│       │                                                                 │
│   75% ┤      ████                                                       │
│       │      ████                                                       │
│   50% ┤      ████     ████                                              │
│       │  ██  ████     ████                                              │
│   25% ┤  ██  ████     ████     ██                            ██         │
│       │  ██  ████     ████     ██                       ██   ██         │
│    0% ┼──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴──   │
│       0%  10%  20%  30%  40%  50%  60%  70%  80%  90%  100%             │
│                       └───────────────┘                                 │
│                        UNCERTAINTY ZONE                                 │
│                                                                         │
│  Summary: LOW: 25% │ BORDERLINE: 35% │ HIGH: 40%                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 Distribution Time Series Panel (Third Row)

```
┌───────────────────────────────────────┬─────────────────────────────────┐
│  PROBABILITY BUCKETS OVER TIME        │  BORDERLINE RATE OVER TIME      │
│                                       │                                 │
│  100%│ ─── HIGH                       │  60%│                           │
│      │ ─── BORDERLINE                 │     │      ╱╲                   │
│   75%│ ─── LOW                        │  40%│─────╱──╲───────────       │
│      │      ──────────                │     │    ╱    ╲  ╱╲             │
│   50%│─────────────────               │  20%│───╱──────╲╱──╲────        │
│      │  ───────────────               │     │                           │
│   25%│                                │   0%│───────────────────        │
│      │                                │     └────────────────────       │
│      └─────────────────────           │       6h    12h    18h    24h   │
│        6h    12h    18h    24h        │                                 │
└───────────────────────────────────────┴─────────────────────────────────┘
```

---

### 4.4 Confidence Distribution Panel (Fourth Row)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CONFIDENCE INTERVAL WIDTH DISTRIBUTION                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  NARROW (<0.10)  ████████████████████████████████████ 72%        │  │
│  │  MEDIUM (0.10-0.20) █████████████ 24%                            │  │
│  │  WIDE (>0.20)    ██ 4%                                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Alert: ⚠️ Wide CI rate increased 2% from baseline                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 4.5 Error Analysis Panel (Fifth Row)

```
┌───────────────────────────────────────┬─────────────────────────────────┐
│  ERROR BREAKDOWN (Last 24h)           │  ERROR TREND                    │
│                                       │                                 │
│  FEATURE_EXTRACTION_ERROR   ███ 45%   │   5%│                           │
│  IMAGE_DECODE_ERROR         ██  30%   │     │    ╱╲                     │
│  INVALID_CONTENT_TYPE       █   15%   │   3%│───╱──╲────────            │
│  TIMEOUT                    ░   5%    │     │                           │
│  OTHER                      ░   5%    │   1%│─────────────────          │
│                                       │     └────────────────           │
│  Total Errors: 234                    │       6h   12h   18h   24h      │
└───────────────────────────────────────┴─────────────────────────────────┘
```

---

### 4.6 Traffic Anomalies Panel (Sixth Row)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRAFFIC ANOMALIES                                                       │
│                                                                         │
│  High-Rate IPs (>100 req/min):                                          │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  IP Hash          │ Rate (1min) │ Last Seen   │ Pattern            ││
│  │  sha256:abc12...  │ 145 req     │ 2min ago    │ Regular interval   ││
│  │  sha256:def34...  │ 112 req     │ 5min ago    │ Burst              ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  ⚠️ 2 IPs flagged for review (no automatic action taken)               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 4.7 Semantic Integrity Panel (Bottom Row)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SEMANTIC INTEGRITY                                                      │
│                                                                         │
│  ✅ Model Version: 1.0.0 (consistent)                                   │
│  ✅ Semantic Version: 1.0.0 (consistent)                                │
│  ✅ API Version: 1.0.0 (consistent)                                     │
│  ✅ Missing Version Metadata: 0                                         │
│  ✅ Interpretation Distribution: Normal                                 │
│                                                                         │
│  Interpretation Breakdown:                                              │
│  LOW                    ████████ 25%                                    │
│  MODERATE-LOW           ███ 8%                                          │
│  INCONCLUSIVE           ████████████ 35%                                │
│  MODERATE-HIGH          ██████ 18%                                      │
│  HIGH                   █████ 14%                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. DASHBOARD SECTIONS SUMMARY

| Panel | Purpose | Refresh Rate |
|-------|---------|--------------|
| Overview | Quick health status | 10 seconds |
| Probability Distribution | Current output distribution | 1 minute |
| Distribution Time Series | Drift detection | 5 minutes |
| Confidence Distribution | Uncertainty monitoring | 5 minutes |
| Error Analysis | Failure concentration | 1 minute |
| Traffic Anomalies | Misuse signals | 1 minute |
| Semantic Integrity | Version/behavior consistency | 1 minute |

---

## 6. WHAT IS NOT INCLUDED

| Excluded | Reason |
|----------|--------|
| User-level drill-downs | Privacy |
| Image content analysis | Privacy |
| User session tracking | Privacy |
| Conversion metrics | Not safety monitoring |
| Engagement metrics | Not safety monitoring |
| A/B testing metrics | Not safety monitoring |
| Marketing attribution | Not safety monitoring |

---

**⛔ STOPPED** — Awaiting final phase prompt.  
No deployment, no cloud provider specifics.
