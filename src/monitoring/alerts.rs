//! Alert System
//!
//! Alerting based on metric thresholds and conditions.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Instant;

/// Severity level for alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Informational
    Info,
    /// Warning condition
    Warning,
    /// Error condition
    Error,
    /// Critical condition requiring immediate attention
    Critical,
}

/// Condition that triggers an alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Metric exceeds threshold
    GreaterThan(f64),
    /// Metric falls below threshold
    LessThan(f64),
    /// Metric equals value (within f64::EPSILON)
    EqualTo(f64),
    /// Metric is outside the given range [low, high]
    OutOfRange(f64, f64),
}

impl AlertCondition {
    /// Evaluate the condition against a metric value
    pub fn evaluate(&self, value: f64) -> bool {
        match self {
            AlertCondition::GreaterThan(threshold) => value > *threshold,
            AlertCondition::LessThan(threshold) => value < *threshold,
            AlertCondition::EqualTo(target) => (value - target).abs() < f64::EPSILON,
            AlertCondition::OutOfRange(low, high) => value < *low || value > *high,
        }
    }
}

/// A single alert definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique name for this alert
    pub name: String,
    /// Name of the metric to monitor
    pub metric_name: String,
    /// Condition that triggers the alert
    pub condition: AlertCondition,
    /// Severity level
    pub level: AlertLevel,
    /// Human-readable message when triggered
    pub message: String,
    /// Minimum seconds between consecutive triggers
    pub cooldown_secs: u64,
    /// Last time this alert was triggered
    #[serde(skip)]
    pub last_triggered: Option<Instant>,
    /// Number of times this alert has been triggered
    pub triggered_count: u64,
}

impl Alert {
    /// Create a new alert
    pub fn new(
        name: impl Into<String>,
        metric_name: impl Into<String>,
        condition: AlertCondition,
        level: AlertLevel,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            metric_name: metric_name.into(),
            condition,
            level,
            message: message.into(),
            cooldown_secs: 60,
            last_triggered: None,
            triggered_count: 0,
        }
    }

    /// Set the cooldown period in seconds
    pub fn with_cooldown(mut self, secs: u64) -> Self {
        self.cooldown_secs = secs;
        self
    }

    /// Check if the alert is within its cooldown period
    pub fn is_in_cooldown(&self) -> bool {
        if let Some(last) = self.last_triggered {
            last.elapsed().as_secs() < self.cooldown_secs
        } else {
            false
        }
    }
}

/// Record of a triggered alert for history tracking
#[derive(Debug, Clone)]
pub struct AlertRecord {
    /// Alert name
    pub name: String,
    /// Severity level
    pub level: AlertLevel,
    /// When the alert was triggered
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// The metric value that triggered the alert
    pub metric_value: f64,
    /// The alert message
    pub message: String,
}

/// Manages alert definitions, evaluation, and history
pub struct AlertManager {
    alerts: Vec<Alert>,
    handlers: Vec<Box<dyn Fn(&Alert, f64) + Send + Sync>>,
    history: Vec<AlertRecord>,
}

impl AlertManager {
    /// Maximum number of alert records retained in history before the
    /// oldest entries are evicted, to prevent unbounded memory growth on a
    /// long-running monitor with a flapping metric.
    const MAX_HISTORY: usize = 1000;

    /// Create a new empty AlertManager
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            handlers: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Add an alert definition
    pub fn add_alert(&mut self, alert: Alert) {
        self.alerts.push(alert);
    }

    /// Remove an alert by name
    pub fn remove_alert(&mut self, name: &str) {
        self.alerts.retain(|a| a.name != name);
    }

    /// Register a handler function called when any alert triggers
    pub fn add_handler(&mut self, handler: Box<dyn Fn(&Alert, f64) + Send + Sync>) {
        self.handlers.push(handler);
    }

    /// Evaluate all alerts against current metric values.
    /// Returns a list of (alert_name, level, message) for each triggered alert.
    pub fn check_alerts(
        &mut self,
        metrics: &HashMap<String, f64>,
    ) -> Vec<(String, AlertLevel, String)> {
        let mut triggered = Vec::new();

        for alert in self.alerts.iter_mut() {
            if let Some(&value) = metrics.get(&alert.metric_name) {
                if alert.condition.evaluate(value) && !alert.is_in_cooldown() {
                    alert.last_triggered = Some(Instant::now());
                    alert.triggered_count += 1;

                    let message = format!("{} (value: {:.4})", alert.message, value);
                    triggered.push((alert.name.clone(), alert.level, message.clone()));

                    self.history.push(AlertRecord {
                        name: alert.name.clone(),
                        level: alert.level,
                        timestamp: chrono::Utc::now(),
                        metric_value: value,
                        message: message.clone(),
                    });
                    if self.history.len() > Self::MAX_HISTORY {
                        let excess = self.history.len() - Self::MAX_HISTORY;
                        self.history.drain(0..excess);
                    }

                    // Invoke handlers — borrow alert immutably via snapshot
                    for handler in &self.handlers {
                        handler(alert, value);
                    }
                }
            }
        }

        triggered
    }

    /// Get references to all currently defined alerts
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.alerts.iter().collect()
    }

    /// Get the history of triggered alerts
    pub fn get_alert_history(&self) -> Vec<(String, AlertLevel, chrono::DateTime<chrono::Utc>)> {
        self.history
            .iter()
            .map(|r| (r.name.clone(), r.level, r.timestamp))
            .collect()
    }

    /// Get full alert history records
    pub fn get_alert_history_full(&self) -> &[AlertRecord] {
        &self.history
    }

    /// Clear alert history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Reset triggered counts and cooldowns on all alerts
    pub fn reset_alerts(&mut self) {
        for alert in &mut self.alerts {
            alert.triggered_count = 0;
            alert.last_triggered = None;
        }
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_condition_greater_than() {
        let cond = AlertCondition::GreaterThan(90.0);
        assert!(cond.evaluate(95.0));
        assert!(!cond.evaluate(85.0));
    }

    #[test]
    fn test_alert_condition_less_than() {
        let cond = AlertCondition::LessThan(10.0);
        assert!(cond.evaluate(5.0));
        assert!(!cond.evaluate(15.0));
    }

    #[test]
    fn test_alert_condition_out_of_range() {
        let cond = AlertCondition::OutOfRange(20.0, 80.0);
        assert!(cond.evaluate(10.0));
        assert!(cond.evaluate(90.0));
        assert!(!cond.evaluate(50.0));
    }

    #[test]
    fn test_alert_manager_check() {
        let mut manager = AlertManager::new();
        manager.add_alert(
            Alert::new(
                "high_cpu",
                "cpu_usage",
                AlertCondition::GreaterThan(90.0),
                AlertLevel::Critical,
                "CPU usage is critically high",
            )
            .with_cooldown(0),
        );

        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 95.0);

        let triggered = manager.check_alerts(&metrics);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].0, "high_cpu");

        let history = manager.get_alert_history();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_alert_cooldown() {
        let mut manager = AlertManager::new();
        manager.add_alert(
            Alert::new(
                "high_mem",
                "memory_usage",
                AlertCondition::GreaterThan(80.0),
                AlertLevel::Warning,
                "Memory usage high",
            )
            .with_cooldown(3600),
        );

        let mut metrics = HashMap::new();
        metrics.insert("memory_usage".to_string(), 90.0);

        let first = manager.check_alerts(&metrics);
        assert_eq!(first.len(), 1);

        // Second check should be suppressed by cooldown
        let second = manager.check_alerts(&metrics);
        assert_eq!(second.len(), 0);
    }

    #[test]
    fn test_remove_alert() {
        let mut manager = AlertManager::new();
        manager.add_alert(Alert::new(
            "test",
            "metric",
            AlertCondition::GreaterThan(0.0),
            AlertLevel::Info,
            "test alert",
        ));

        assert_eq!(manager.get_active_alerts().len(), 1);
        manager.remove_alert("test");
        assert_eq!(manager.get_active_alerts().len(), 0);
    }
}
