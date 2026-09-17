//! URL-based dataset import with auto-detection of source type

use tracing::{info, warn};
use std::net::IpAddr;

/// Build an HTTP client with SSRF-safe redirect policy. When `pin` is provided, the
/// client is forced to connect to that exact (host, socket address) pair instead of
/// performing its own DNS resolution — this closes a DNS-rebinding TOCTOU where a
/// low-TTL attacker-controlled domain could resolve to a public IP during
/// `validate_url_safety` but to a private/internal IP (e.g. 169.254.169.254,
/// 127.0.0.1) when reqwest independently re-resolves it to make the actual request.
fn safe_http_client(timeout_secs: u64) -> Result<reqwest::Client, String> {
    safe_http_client_pinned(timeout_secs, None)
}

fn safe_http_client_pinned(
    timeout_secs: u64,
    pin: Option<(&str, std::net::SocketAddr)>,
) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            // Redirect targets are re-validated here on every hop (kept as-is); the
            // initial connection's IP pinning is handled separately via `.resolve()`.
            if attempt.previous().len() >= 5 {
                attempt.error("too many redirects")
            } else if let Some(host) = attempt.url().host_str() {
                let lower = host.to_lowercase();
                if lower == "localhost" || lower.ends_with(".local")
                    || lower.ends_with(".internal") || lower == "metadata.google.internal"
                {
                    attempt.error("redirect to internal host blocked")
                } else if let Ok(ip) = host.parse::<IpAddr>() {
                    if is_private_ip(&ip) {
                        attempt.error("redirect to private IP blocked")
                    } else {
                        attempt.follow()
                    }
                } else {
                    attempt.follow()
                }
            } else {
                attempt.error("redirect has no host")
            }
        }));

    if let Some((host, addr)) = pin {
        builder = builder.resolve(host, addr);
    }

    builder
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))
}

/// Resolve `host` to a concrete, validated (non-private/non-reserved) socket address so
/// the caller can pin the actual HTTP connection to it. This performs the DNS
/// resolution exactly once and validates every returned address, then returns a single
/// address to connect to — preventing a second, independent DNS lookup (as reqwest
/// would otherwise perform internally) from returning a different, unvalidated IP.
fn resolve_validated_socket_addr(host: &str, port: u16) -> Result<std::net::SocketAddr, String> {
    // Host is already a literal IP - validate directly, no DNS involved.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_private_ip(&ip) {
            return Err("Access to private/reserved IP addresses is not allowed".to_string());
        }
        return Ok(std::net::SocketAddr::new(ip, port));
    }

    let addrs: Vec<std::net::SocketAddr> =
        std::net::ToSocketAddrs::to_socket_addrs(&(host, port))
            .map_err(|e| format!("DNS resolution failed for host '{}': {}", host, e))?
            .collect();

    if addrs.is_empty() {
        return Err(format!("DNS resolution returned no addresses for host '{}'", host));
    }

    for addr in &addrs {
        if is_private_ip(&addr.ip()) {
            return Err("URL resolves to a private/reserved IP address".to_string());
        }
    }

    // Pin to the first validated address; every candidate was checked above, so any of
    // them is safe, but using a single fixed address is what makes pinning effective.
    Ok(addrs[0])
}

/// Check if an IP address is in a private/reserved range (SSRF protection)
fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            ipv4.is_loopback()          // 127.0.0.0/8
                || ipv4.is_private()    // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || ipv4.is_link_local() // 169.254.0.0/16
                || ipv4.is_broadcast()  // 255.255.255.255
                || ipv4.is_unspecified() // 0.0.0.0
                || ipv4.octets()[0] == 100 && (ipv4.octets()[1] & 0xC0) == 64 // 100.64.0.0/10 (CGNAT)
                || ipv4.octets()[0] == 192 && ipv4.octets()[1] == 0 && ipv4.octets()[2] == 0 // 192.0.0.0/24
        }
        IpAddr::V6(ipv6) => {
            ipv6.is_loopback() || ipv6.is_unspecified()
        }
    }
}

/// Validate URL for SSRF attacks — only allow http/https to public IPs
pub fn validate_url_safety(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;

    // Only allow http/https
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("Unsupported URL scheme: '{}'. Only http and https are allowed.", scheme)),
    }

    // Check for IP-based URLs
    if let Some(host) = parsed.host_str() {
        // Block common private hostnames
        let lower_host = host.to_lowercase();
        if lower_host == "localhost" || lower_host == "metadata.google.internal"
            || lower_host.ends_with(".local") || lower_host.ends_with(".internal")
        {
            return Err("Access to internal/local hosts is not allowed".to_string());
        }

        // If host is an IP, check it's not private
        if let Ok(ip) = host.parse::<IpAddr>() {
            if is_private_ip(&ip) {
                return Err("Access to private/reserved IP addresses is not allowed".to_string());
            }
        }

        // DNS resolution check — resolve hostname and verify all IPs are public
        if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&(host, parsed.port().unwrap_or(80))) {
            for addr in addrs {
                if is_private_ip(&addr.ip()) {
                    return Err("URL resolves to a private/reserved IP address".to_string());
                }
            }
        }
    } else {
        return Err("URL has no host".to_string());
    }

    Ok(())
}

/// Detected source type from a URL
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum SourceType {
    Kaggle { slug: String },
    GitHub { raw_url: String },
    GoogleSheets { csv_url: String },
    HuggingFace { dataset_id: String },
    RawFile { format: FileFormat },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum FileFormat {
    Csv,
    Tsv,
    Json,
    Parquet,
    Excel,
    Unknown,
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceType::Kaggle { slug } => write!(f, "Kaggle ({})", slug),
            SourceType::GitHub { .. } => write!(f, "GitHub"),
            SourceType::GoogleSheets { .. } => write!(f, "Google Sheets"),
            SourceType::HuggingFace { dataset_id } => write!(f, "Hugging Face ({})", dataset_id),
            SourceType::RawFile { format } => write!(f, "Direct URL ({:?})", format),
            SourceType::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::fmt::Display for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileFormat::Csv => write!(f, "csv"),
            FileFormat::Tsv => write!(f, "tsv"),
            FileFormat::Json => write!(f, "json"),
            FileFormat::Parquet => write!(f, "parquet"),
            FileFormat::Excel => write!(f, "excel"),
            FileFormat::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detect source type from URL
pub fn detect_source(url: &str) -> SourceType {
    let url_lower = url.to_lowercase();

    // Kaggle
    if url_lower.contains("kaggle.com/datasets/") || url_lower.contains("kaggle.com/competitions/") {
        let slug = url.split("/datasets/")
            .nth(1)
            .or_else(|| url.split("/competitions/").nth(1))
            .map(|s| {
                // Strip query params and trailing slashes
                s.split('?').next().unwrap_or(s).trim_end_matches('/').to_string()
            })
            .unwrap_or_default();
        return SourceType::Kaggle { slug };
    }

    // GitHub
    if url_lower.contains("github.com/") && url_lower.contains("/blob/") {
        let raw_url = url
            .replace("github.com", "raw.githubusercontent.com")
            .replace("/blob/", "/");
        return SourceType::GitHub { raw_url };
    }
    if url_lower.contains("raw.githubusercontent.com/") {
        return SourceType::GitHub { raw_url: url.to_string() };
    }

    // Google Sheets
    if url_lower.contains("docs.google.com/spreadsheets/") {
        let sheet_id = url.split("/d/")
            .nth(1)
            .and_then(|s| s.split('/').next())
            .unwrap_or("");
        let csv_url = format!("https://docs.google.com/spreadsheets/d/{}/export?format=csv", sheet_id);
        return SourceType::GoogleSheets { csv_url };
    }

    // Hugging Face
    if url_lower.contains("huggingface.co/datasets/") {
        let dataset_id = url.split("/datasets/")
            .nth(1)
            .map(|s| s.split('?').next().unwrap_or(s).trim_end_matches('/').to_string())
            .unwrap_or_default();
        return SourceType::HuggingFace { dataset_id };
    }

    // Raw file URL by extension
    let path = url.split('?').next().unwrap_or(url);
    let format = detect_format_from_path(path);
    if format != FileFormat::Unknown {
        return SourceType::RawFile { format };
    }

    SourceType::Unknown
}

/// Detect file format from path/extension
pub fn detect_format_from_path(path: &str) -> FileFormat {
    let lower = path.to_lowercase();
    if lower.ends_with(".csv") { FileFormat::Csv }
    else if lower.ends_with(".tsv") || lower.ends_with(".tab") { FileFormat::Tsv }
    else if lower.ends_with(".json") || lower.ends_with(".jsonl") || lower.ends_with(".ndjson") { FileFormat::Json }
    else if lower.ends_with(".parquet") || lower.ends_with(".pq") { FileFormat::Parquet }
    else if lower.ends_with(".xlsx") || lower.ends_with(".xls") { FileFormat::Excel }
    else { FileFormat::Unknown }
}

/// Detect format from Content-Type header
pub fn detect_format_from_content_type(ct: &str) -> FileFormat {
    let ct_lower = ct.to_lowercase();
    if ct_lower.contains("text/csv") || ct_lower.contains("text/comma-separated") { FileFormat::Csv }
    else if ct_lower.contains("text/tab-separated") { FileFormat::Tsv }
    else if ct_lower.contains("application/json") || ct_lower.contains("text/json") { FileFormat::Json }
    else if ct_lower.contains("application/parquet") || ct_lower.contains("application/x-parquet") { FileFormat::Parquet }
    else if ct_lower.contains("spreadsheetml") || ct_lower.contains("excel") { FileFormat::Excel }
    else if ct_lower.contains("text/plain") { FileFormat::Csv } // assume CSV for plain text
    else { FileFormat::Unknown }
}

/// Download dataset from a URL. Returns (bytes, detected_format, filename).
pub async fn download_dataset(url: &str, source: &SourceType) -> Result<(Vec<u8>, FileFormat, String), String> {
    const MAX_SIZE: usize = 500 * 1024 * 1024; // 500MB limit

    // SSRF protection: validate URL before making any request
    validate_url_safety(url)?;

    let download_url = match source {
        SourceType::GitHub { raw_url } => raw_url.clone(),
        SourceType::GoogleSheets { csv_url } => csv_url.clone(),
        SourceType::RawFile { .. } | SourceType::Unknown => url.to_string(),
        SourceType::Kaggle { slug } => {
            // Try Kaggle API if credentials available
            return download_kaggle(slug).await;
        }
        SourceType::HuggingFace { dataset_id } => {
            return download_huggingface(dataset_id).await;
        }
    };

    info!(url = %download_url, source = %source, "Downloading dataset");

    // Re-resolve and re-validate the actual download host right before connecting, and
    // pin the connection to that exact validated IP. Without this, `validate_url_safety`
    // above and reqwest's own internal DNS lookup during `.send()` are two independent
    // resolutions of the same hostname; a low-TTL DNS record could answer safely for the
    // first and point at a private/internal address (e.g. 169.254.169.254) for the
    // second, bypassing the SSRF guard entirely (DNS-rebinding TOCTOU).
    let parsed_download_url = url::Url::parse(&download_url)
        .map_err(|e| format!("Invalid download URL: {}", e))?;
    let download_host = parsed_download_url.host_str()
        .ok_or_else(|| "Download URL has no host".to_string())?
        .to_string();
    let download_port = parsed_download_url.port_or_known_default().unwrap_or(443);
    let pinned_addr = resolve_validated_socket_addr(&download_host, download_port)?;

    let client = safe_http_client_pinned(300, Some((&download_host, pinned_addr)))?;

    let response = client.get(&download_url)
        .header("User-Agent", "AutoML/0.5")
        .send()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP error {}: {}", response.status().as_u16(), response.status().canonical_reason().unwrap_or("Unknown")));
    }

    // Check content length
    if let Some(len) = response.content_length() {
        if len as usize > MAX_SIZE {
            return Err(format!("File too large: {} MB (limit: {} MB)", len / 1024 / 1024, MAX_SIZE / 1024 / 1024));
        }
    }

    // Detect format from content-type header
    let content_type = response.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Get filename from Content-Disposition or URL
    let filename = response.headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.split("filename=").nth(1)
                .map(|s| s.trim_matches('"').trim_matches('\'').to_string())
        })
        .unwrap_or_else(|| {
            let path = download_url.split('?').next().unwrap_or(&download_url);
            path.split('/').last().unwrap_or("data.csv").to_string()
        });

    let bytes = response.bytes().await
        .map_err(|e| format!("Failed to read response body: {}", e))?;

    if bytes.len() > MAX_SIZE {
        return Err(format!("File too large: {} MB (limit: {} MB)", bytes.len() / 1024 / 1024, MAX_SIZE / 1024 / 1024));
    }

    // Determine format: try URL path first, then content-type, then sniff
    let format = match source {
        SourceType::GoogleSheets { .. } => FileFormat::Csv,
        SourceType::RawFile { format } => format.clone(),
        _ => {
            let from_path = detect_format_from_path(&filename);
            if from_path != FileFormat::Unknown {
                from_path
            } else {
                let from_ct = detect_format_from_content_type(&content_type);
                if from_ct != FileFormat::Unknown {
                    from_ct
                } else {
                    sniff_format(&bytes)
                }
            }
        }
    };

    info!(
        size_bytes = bytes.len(),
        format = %format,
        filename = %filename,
        "Dataset downloaded successfully"
    );

    Ok((bytes.to_vec(), format, filename))
}

/// Sniff file format from first bytes
fn sniff_format(bytes: &[u8]) -> FileFormat {
    if bytes.len() < 4 {
        return FileFormat::Unknown;
    }
    // Parquet magic bytes: PAR1
    if bytes.starts_with(b"PAR1") {
        return FileFormat::Parquet;
    }
    // JSON starts with { or [
    let trimmed = bytes.iter().position(|&b| !b.is_ascii_whitespace());
    if let Some(pos) = trimmed {
        if bytes[pos] == b'{' || bytes[pos] == b'[' {
            return FileFormat::Json;
        }
    }
    // Excel: ZIP header (PK)
    if bytes.starts_with(&[0x50, 0x4B]) {
        return FileFormat::Excel;
    }
    // Default to CSV
    FileFormat::Csv
}

/// Download from Kaggle API (requires KAGGLE_USERNAME + KAGGLE_KEY env vars)
async fn download_kaggle(slug: &str) -> Result<(Vec<u8>, FileFormat, String), String> {
    let username = std::env::var("KAGGLE_USERNAME")
        .map_err(|_| "KAGGLE_USERNAME env var not set. Set KAGGLE_USERNAME and KAGGLE_KEY to enable direct Kaggle downloads.".to_string())?;
    let key = std::env::var("KAGGLE_KEY")
        .map_err(|_| "KAGGLE_KEY env var not set. Set KAGGLE_USERNAME and KAGGLE_KEY to enable direct Kaggle downloads.".to_string())?;

    let api_url = format!("https://www.kaggle.com/api/v1/datasets/download/{}", slug);
    info!(slug = %slug, "Downloading from Kaggle API");

    let client = safe_http_client(600)?;

    let response = client.get(&api_url)
        .basic_auth(&username, Some(&key))
        .header("User-Agent", "AutoML/0.5")
        .send()
        .await
        .map_err(|e| format!("Kaggle API request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Kaggle API error {}: check credentials and dataset slug '{}'", response.status(), slug));
    }

    let bytes = response.bytes().await
        .map_err(|e| format!("Failed to read Kaggle response: {}", e))?;

    let filename = format!("{}.zip", slug.replace('/', "_"));
    // Kaggle downloads are typically ZIP files containing CSV
    Ok((bytes.to_vec(), FileFormat::Csv, filename))
}

/// Download from Hugging Face datasets
async fn download_huggingface(dataset_id: &str) -> Result<(Vec<u8>, FileFormat, String), String> {
    // Try parquet first (common on HF), then CSV
    let parquet_url = format!(
        "https://huggingface.co/datasets/{}/resolve/main/data/train-00000-of-00001.parquet",
        dataset_id
    );
    let csv_url = format!(
        "https://huggingface.co/datasets/{}/resolve/main/data/train.csv",
        dataset_id
    );

    let client = safe_http_client(300)?;

    // Try parquet
    info!(dataset_id = %dataset_id, "Trying HuggingFace parquet download");
    let resp = client.get(&parquet_url)
        .header("User-Agent", "AutoML/0.5")
        .send().await;
    if let Ok(r) = resp {
        if r.status().is_success() {
            let bytes = r.bytes().await.map_err(|e| format!("HF read error: {}", e))?;
            let filename = format!("{}_train.parquet", dataset_id.replace('/', "_"));
            return Ok((bytes.to_vec(), FileFormat::Parquet, filename));
        }
    }

    // Try CSV
    warn!(dataset_id = %dataset_id, "Parquet not found, trying CSV");
    let resp = client.get(&csv_url)
        .header("User-Agent", "AutoML/0.5")
        .send().await
        .map_err(|e| format!("HF download failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Could not download HuggingFace dataset '{}'. Try a direct file URL instead.",
            dataset_id
        ));
    }

    let bytes = resp.bytes().await.map_err(|e| format!("HF read error: {}", e))?;
    let filename = format!("{}_train.csv", dataset_id.replace('/', "_"));
    Ok((bytes.to_vec(), FileFormat::Csv, filename))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_kaggle() {
        let s = detect_source("https://www.kaggle.com/datasets/uciml/iris");
        assert!(matches!(s, SourceType::Kaggle { slug } if slug == "uciml/iris"));
    }

    #[test]
    fn test_detect_github() {
        let s = detect_source("https://github.com/user/repo/blob/main/data.csv");
        assert!(matches!(s, SourceType::GitHub { raw_url } if raw_url.contains("raw.githubusercontent.com")));
    }

    #[test]
    fn test_detect_google_sheets() {
        let s = detect_source("https://docs.google.com/spreadsheets/d/1abc123/edit#gid=0");
        assert!(matches!(s, SourceType::GoogleSheets { .. }));
    }

    #[test]
    fn test_detect_raw_csv() {
        let s = detect_source("https://example.com/data/file.csv");
        assert!(matches!(s, SourceType::RawFile { format: FileFormat::Csv }));
    }

    #[test]
    fn test_detect_raw_parquet() {
        let s = detect_source("https://example.com/data/file.parquet?version=1");
        assert!(matches!(s, SourceType::RawFile { format: FileFormat::Parquet }));
    }

    #[test]
    fn test_detect_huggingface() {
        let s = detect_source("https://huggingface.co/datasets/scikit-learn/iris");
        assert!(matches!(s, SourceType::HuggingFace { dataset_id } if dataset_id == "scikit-learn/iris"));
    }

    #[test]
    fn test_sniff_parquet() {
        assert_eq!(sniff_format(b"PAR1\x00\x00"), FileFormat::Parquet);
    }

    #[test]
    fn test_sniff_json() {
        assert_eq!(sniff_format(b"  {\"key\": \"value\"}"), FileFormat::Json);
    }

    #[test]
    fn test_sniff_csv() {
        assert_eq!(sniff_format(b"col1,col2,col3\n1,2,3"), FileFormat::Csv);
    }
}
