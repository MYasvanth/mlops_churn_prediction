# alert_system.py
"""
Alert System Module

This module provides a comprehensive alert system for ML model monitoring,
including email notifications, Slack integration, and alert management.
"""

import smtplib
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import logging
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Alert types"""
    DATA_DRIFT = "DATA_DRIFT"
    MODEL_DEGRADATION = "MODEL_DEGRADATION"
    THRESHOLD_BREACH = "THRESHOLD_BREACH"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    INFRASTRUCTURE = "INFRASTRUCTURE"

@dataclass
class Alert:
    """Data class for alert information"""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

class EmailNotifier:
    """Email notification service"""
    
    def __init__(self, config: Dict):
        """
        Initialize email notifier
        
        Args:
            config: Email configuration dictionary
        """
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.use_tls = config.get('use_tls', True)
        
        self.enabled = all([
            self.smtp_server, self.username, 
            self.password, self.from_email
        ])
        
        if not self.enabled:
            logger.warning("Email notifier not properly configured")
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """
        Send alert via email
        
        Args:
            alert: Alert object
            recipients: List of recipient email addresses
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Email notifier not enabled")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """
        Create HTML email body
        
        Args:
            alert: Alert object
            
        Returns:
            HTML formatted email body
        """
        severity_colors = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .alert-header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .alert-content {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
                .metadata {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; }}
                .timestamp {{ color: #6c757d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🚨 {alert.title}</h2>
                <p>Severity: {alert.severity.value} | Type: {alert.alert_type.value}</p>
            </div>
            
            <div class="alert-content">
                <h3>Alert Details</h3>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p class="timestamp"><strong>Timestamp:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                
                {self._format_metadata_html(alert.metadata) if alert.metadata else ''}
            </div>
            
            <div class="metadata">
                <p><strong>Alert ID:</strong> {alert.id}</p>
                <p><em>This is an automated alert from the ML Model Monitoring System</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _format_metadata_html(self, metadata: Dict) -> str:
        """Format metadata as HTML"""
        if not metadata:
            return ""
        
        html = "<div class='metadata'><h4>Additional Information</h4>"
        for key, value in metadata.items():
            html += f"<p><strong>{key}:</strong> {value}</p>"
        html += "</div>"
        
        return html

class SlackNotifier:
    """Slack notification service"""
    
    def __init__(self, config: Dict):
        """
        Initialize Slack notifier
        
        Args:
            config: Slack configuration dictionary
        """
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'ML Monitor Bot')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
        
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Slack notifier not properly configured")
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to Slack
        
        Args:
            alert: Alert object
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier not enabled")
            return False
        
        try:
            # Create Slack message
            message = self._create_slack_message(alert)
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
                return True
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _create_slack_message(self, alert: Alert) -> Dict:
        """
        Create Slack message payload
        
        Args:
            alert: Alert object
            
        Returns:
            Slack message dictionary
        """
        severity_colors = {
            AlertSeverity.LOW: 'good',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.HIGH: 'danger',
            AlertSeverity.CRITICAL: 'danger'
        }
        
        color = severity_colors.get(alert.severity, 'good')
        
        # Create fields for metadata
        fields = [
            {
                "title": "Source",
                "value": alert.source,
                "short": True
            },
            {
                "title": "Severity",
                "value": alert.severity.value,
                "short": True
            },
            {
                "title": "Type",
                "value": alert.alert_type.value,
                "short": True
            },
            {
                "title": "Timestamp",
                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                "short": True
            }
        ]
        
        # Add metadata fields
        if alert.metadata:
            for key, value in alert.metadata.items():
                fields.append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
        
        message = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": fields,
                    "footer": f"Alert ID: {alert.id}",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        return message

class AlertManager:
    """Main alert management system"""
    
    def __init__(self, config_path: str = "configs/monitoring/monitoring_config.yaml"):
        """
        Initialize alert manager
        
        Args:
            config_path: Path to monitoring configuration file
        """
        self.config = load_config(config_path)
        self.alert_config = self.config.get('alerts', {})
        
        # Initialize notifiers
        self.email_notifier = EmailNotifier(
            self.alert_config.get('email', {})
        )
        self.slack_notifier = SlackNotifier(
            self.alert_config.get('slack', {})
        )
        
        # Alert storage
        self.alerts: List[Alert] = []
        self.alert_rules = self.alert_config.get('rules', {})
        
        # Rate limiting
        self.rate_limits = self.alert_config.get('rate_limits', {})
        self.last_alert_times = {}
        
        logger.info("AlertManager initialized")
    
    def create_alert(self,
                    alert_type: AlertType,
                    severity: AlertSeverity,
                    title: str,
                    message: str,
                    source: str,
                    metadata: Optional[Dict] = None) -> Alert:
        """
        Create a new alert
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            title: Alert title
            message: Alert message
            source: Source of the alert
            metadata: Additional metadata
            
        Returns:
            Created Alert object
        """
        alert_id = f"{alert_type.value}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.info(f"Alert created: {alert.id}")
        
        return alert
    
    def should_send_alert(self, alert: Alert) -> bool:
        """
        Check if alert should be sent based on rate limiting and rules
        
        Args:
            alert: Alert object
            
        Returns:
            True if alert should be sent, False otherwise
        """
        # Check rate limiting
        rate_limit_key = f"{alert.alert_type.value}_{alert.source}"
        rate_limit_minutes = self.rate_limits.get(
            alert.alert_type.value, 
            self.rate_limits.get('default', 15)
        )
        
        if rate_limit_key in self.last_alert_times:
            last_time = self.last_alert_times[rate_limit_key]
            time_diff = datetime.now() - last_time
            if time_diff.total_seconds() < rate_limit_minutes * 60:
                logger.info(f"Alert rate limited: {alert.id}")
                return False
        
        # Check alert rules
        if not self._check_alert_rules(alert):
            logger.info(f"Alert filtered by rules: {alert.id}")
            return False
        
        return True
    
    def _check_alert_rules(self, alert: Alert) -> bool:
        """
        Check if alert passes configured rules
        
        Args:
            alert: Alert object
            
        Returns:
            True if alert passes rules, False otherwise
        """
        # Check severity threshold
        min_severity = self.alert_rules.get('min_severity', 'LOW')
        severity_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        if severity_order.index(alert.severity.value) < severity_order.index(min_severity):
            return False
        
        # Check business hours (if configured)
        business_hours = self.alert_rules.get('business_hours_only', False)
        if business_hours:
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 17:  # Outside 9 AM - 5 PM
                return False
        
        # Check alert type filters
        disabled_types = self.alert_rules.get('disabled_types', [])
        if alert.alert_type.value in disabled_types:
            return False
        
        return True
    
    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert through configured channels
        
        Args:
            alert: Alert object
            
        Returns:
            Dictionary with send status for each channel
        """
        results = {}
        
        if not self.should_send_alert(alert):
            return {'skipped': True}
        
        try:
            # Send email notifications
            if self.email_notifier.enabled:
                recipients = self.alert_config.get('email', {}).get('recipients', [])
                if recipients:
                    results['email'] = self.email_notifier.send_alert(alert, recipients)
                else:
                    logger.warning("No email recipients configured")
                    results['email'] = False
            
            # Send Slack notifications
            if self.slack_notifier.enabled:
                results['slack'] = self.slack_notifier.send_alert(alert)
            
            # Update rate limiting
            rate_limit_key = f"{alert.alert_type.value}_{alert.source}"
            self.last_alert_times[rate_limit_key] = datetime.now()
            
            logger.info(f"Alert sent: {alert.id}, Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error sending alert {alert.id}: {str(e)}")
            return {'error': str(e)}
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """
        Mark an alert as resolved
        
        Args:
            alert_id: Alert ID to resolve
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved, False if not found
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolved_by = resolved_by
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                return True
        
        logger.warning(f"Alert not found for resolution: {alert_id}")
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active (unresolved) alerts
        
        Returns:
            List of active alerts
        """
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """
        Get alerts by severity level
        
        Args:
            severity: Alert severity level
            
        Returns:
            List of alerts with specified severity
        """
        return [alert for alert in self.alerts if alert.severity == severity]
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """
        Get alerts by type
        
        Args:
            alert_type: Alert type
            
        Returns:
            List of alerts with specified type
        """
        return [alert for alert in self.alerts if alert.alert_type == alert_type]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get alerts from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_alert_statistics(self) -> Dict:
        """
        Get alert statistics
        
        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(self.get_alerts_by_severity(severity))
        
        # Count by type
        type_counts = {}
        for alert_type in AlertType:
            type_counts[alert_type.value] = len(self.get_alerts_by_type(alert_type))
        
        # Recent alerts
        recent_24h = len(self.get_recent_alerts(24))
        recent_7d = len(self.get_recent_alerts(24 * 7))
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': total_alerts - active_alerts,
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'recent_alerts': {
                'last_24h': recent_24h,
                'last_7d': recent_7d
            }
        }
    
    def export_alerts(self, output_path: str = "reports/alerts/") -> str:
        """
        Export alerts to JSON file
        
        Args:
            output_path: Output directory path
            
        Returns:
            Path to exported file
        """
        try:
            import os
            os.makedirs(output_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alerts_export_{timestamp}.json"
            filepath = os.path.join(output_path, filename)
            
            # Convert alerts to serializable format
            alerts_data = []
            for alert in self.alerts:
                alert_dict = asdict(alert)
                alert_dict['timestamp'] = alert.timestamp.isoformat()
                if alert.resolved_at:
                    alert_dict['resolved_at'] = alert.resolved_at.isoformat()
                alert_dict['alert_type'] = alert.alert_type.value
                alert_dict['severity'] = alert.severity.value
                alerts_data.append(alert_dict)
            
            with open(filepath, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
            
            logger.info(f"Alerts exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting alerts: {str(e)}")
            raise
    
    def create_and_send_alert(self,
                             alert_type: AlertType,
                             severity: AlertSeverity,
                             title: str,
                             message: str,
                             source: str,
                             metadata: Optional[Dict] = None) -> Dict:
        """
        Create and send an alert in one operation
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            title: Alert title
            message: Alert message
            source: Source of the alert
            metadata: Additional metadata
            
        Returns:
            Dictionary with alert creation and send results
        """
        try:
            # Create alert
            alert = self.create_alert(
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                source=source,
                metadata=metadata
            )
            
            # Send alert
            send_results = self.send_alert(alert)
            
            return {
                'alert_id': alert.id,
                'created': True,
                'send_results': send_results
            }
            
        except Exception as e:
            logger.error(f"Error creating and sending alert: {str(e)}")
            return {
                'created': False,
                'error': str(e)
            }
    
    def cleanup_old_alerts(self, days: int = 30) -> int:
        """
        Remove alerts older than specified days
        
        Args:
            days: Number of days to keep alerts
            
        Returns:
            Number of alerts removed
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        initial_count = len(self.alerts)
        
        self.alerts = [
            alert for alert in self.alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        removed_count = initial_count - len(self.alerts)
        logger.info(f"Cleaned up {removed_count} old alerts")
        
        return removed_count
    
    def test_notifications(self) -> Dict:
        """
        Test notification channels
        
        Returns:
            Dictionary with test results
        """
        test_alert = self.create_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.LOW,
            title="Test Alert",
            message="This is a test alert to verify notification channels",
            source="alert_system_test",
            metadata={'test': True}
        )
        
        results = self.send_alert(test_alert)
        
        # Mark test alert as resolved
        self.resolve_alert(test_alert.id, "system_test")
        
        return {
            'test_alert_id': test_alert.id,
            'notification_results': results
        }