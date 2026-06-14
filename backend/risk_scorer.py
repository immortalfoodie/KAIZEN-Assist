"""
KAIZEN — Risk Scorer (Pillar 3)
Pure-Python Isolation Forest anomaly detection + weighted score combination.
No numpy or scikit-learn dependencies.
"""

import logging
import math
import random
from datetime import datetime
from typing import Dict, Any, List
from models import AgentAction, RuleViolation, MemoryWarning
from config import settings

logger = logging.getLogger(__name__)


def clip(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))


def poisson(lam: float) -> int:
    """Generate a random Poisson variable using Knuth's algorithm."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


# ── Simple Isolation Forest ──────────────────────────────────────────────────

class IsolationTreeNode:
    def __init__(self, left=None, right=None, split_feature=None, split_value=None, size=0):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value
        self.size = size


def c_factor(n: int) -> float:
    """Average path length of unsuccessful searches in a Binary Search Tree."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # Euler-Mascheroni constant approximation: 0.5772156649
    return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


class IsolationTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: List[List[float]], depth=0) -> IsolationTreeNode:
        n_samples = len(X)
        if n_samples <= 1 or depth >= self.max_depth:
            return IsolationTreeNode(size=n_samples)

        # Check if all samples are identical
        first = X[0]
        if all(x == first for x in X):
            return IsolationTreeNode(size=n_samples)

        n_features = len(first)
        possible_features = list(range(n_features))
        random.shuffle(possible_features)

        split_feature = None
        split_value = None
        for f in possible_features:
            values = [x[f] for x in X]
            min_val = min(values)
            max_val = max(values)
            if min_val < max_val:
                split_feature = f
                split_value = random.uniform(min_val, max_val)
                break

        if split_feature is None:
            return IsolationTreeNode(size=n_samples)

        left_X = [x for x in X if x[split_feature] < split_value]
        right_X = [x for x in X if x[split_feature] >= split_value]

        node = IsolationTreeNode(
            split_feature=split_feature,
            split_value=split_value,
            size=n_samples
        )
        node.left = self.fit(left_X, depth + 1)
        node.right = self.fit(right_X, depth + 1)
        return node

    def path_length(self, x: List[float], node: IsolationTreeNode, depth=0) -> float:
        if node is None:
            return depth
        if node.left is None or node.right is None:
            return depth + c_factor(node.size)

        if x[node.split_feature] < node.split_value:
            return self.path_length(x, node.left, depth + 1)
        else:
            return self.path_length(x, node.right, depth + 1)


class SimpleIsolationForest:
    """Pure-Python Isolation Forest anomaly detector."""
    def __init__(self, n_estimators=100, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.trees: List[IsolationTree] = []
        self.c = 1.0
        self.threshold = 0.5

    def fit(self, X: List[List[float]]):
        n_samples = len(X)
        if n_samples <= 1:
            return
        
        # Max depth limit in Isolation Forest
        max_depth = int(math.ceil(math.log2(max(n_samples, 2))))
        self.c = c_factor(n_samples)
        
        self.trees = []
        for _ in range(self.n_estimators):
            subsample_size = min(256, n_samples)
            subsample = random.sample(X, subsample_size)
            tree = IsolationTree(max_depth=max_depth)
            tree.root = tree.fit(subsample)
            self.trees.append(tree)

        # Compute threshold based on contamination
        scores = [self.score(x) for x in X]
        scores.sort()
        # Find index of threshold corresponding to contamination
        idx = int(len(scores) * (1 - self.contamination))
        if idx >= len(scores):
            idx = len(scores) - 1
        self.threshold = scores[idx]

    def score(self, x: List[float]) -> float:
        if not self.trees:
            return 0.5
        paths = [tree.path_length(x, tree.root) for tree in self.trees]
        avg_path = sum(paths) / len(paths)
        if self.c == 0:
            return 0.5
        return 2.0 ** (- (avg_path / self.c))

    def decision_function(self, x: List[float]) -> float:
        """
        Mimics scikit-learn's decision_function.
        Returns:
            More positive = more normal.
            More negative = more anomalous.
        Range is mapped to ~ [-0.5, 0.5].
        """
        s = self.score(x)
        return 0.5 - s

    def predict(self, x: List[float]) -> int:
        """
        Mimics scikit-learn's predict.
        Returns:
            -1 for anomalous.
            1 for normal.
        """
        s = self.score(x)
        return -1 if s > self.threshold else 1


# ── Risk Scorer ─────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Combines:
      - Custom Isolation Forest anomaly detection (statistical)
      - Rules score (deterministic)
      - Memory score (contextual)
    
    Into a single 0-100 risk score with a decision recommendation.
    """

    def __init__(self):
        self.iso_forest = SimpleIsolationForest(
            contamination=0.1,
            n_estimators=100,
        )
        self._is_trained = False
        self._train_with_seed_data()
        logger.info("RiskScorer initialized with custom pure-Python Isolation Forest")

    def _train_with_seed_data(self):
        """Train anomaly detector with realistic synthetic data using random/math."""
        random.seed(42)
        X_train = []
        
        n_normal = 90
        n_anomalous = 10

        # Normal transactions: low-medium amounts, business hours, weekdays
        for _ in range(n_normal):
            amount = clip(random.lognormvariate(6, 1.2), 50.0, 45000.0)
            frequency = float(clip(poisson(1.5), 0, 5))
            hour = clip(random.gauss(13.0, 2.5), 8.0, 18.0)
            day = float(random.choice([0, 1, 2, 3, 4]))
            tenure = clip(random.gauss(180.0, 90.0), 5.0, 1000.0)
            tool_chain_len = float(random.choices([1, 2], weights=[0.8, 0.2])[0])
            
            X_train.append([amount, frequency, hour, day, tenure, tool_chain_len])

        # Anomalous transactions: high amounts, odd hours, weekends
        for _ in range(n_anomalous):
            amount = clip(random.lognormvariate(10, 1.5), 50000.0, 500000.0)
            frequency = float(clip(poisson(5.0), 3, 10))
            hour = float(random.choice([0, 1, 2, 3, 4, 22, 23]))
            day = float(random.choice([5, 6]))
            tenure = random.uniform(1.0, 30.0)
            tool_chain_len = float(random.choice([3, 4, 5]))
            
            X_train.append([amount, frequency, hour, day, tenure, tool_chain_len])

        self.iso_forest.fit(X_train)
        self._is_trained = True
        logger.info(f"Custom Isolation Forest trained on {len(X_train)} samples")

    def extract_features(self, action: AgentAction) -> List[float]:
        """Convert an action into a numerical feature list."""
        try:
            ts = datetime.fromisoformat(action.timestamp.replace("Z", "+00:00"))
            hour = float(ts.hour)
            day = float(ts.weekday())
        except Exception:
            hour = 12.0
            day = 2.0

        # Estimate frequency from context (default: 1)
        frequency = float(action.context.get("prior_attempts", 1))

        # Customer tenure (default: 30 days)
        tenure = float(action.context.get("customer_tenure_days", 30))

        return [
            float(action.amount),
            frequency,
            hour,
            day,
            tenure,
            1.0,  # tool chain length (simplified)
        ]

    def detect_anomaly(self, action: AgentAction) -> Dict[str, Any]:
        """
        Run Isolation Forest on the action features.
        
        Returns:
            {
                "is_anomaly": bool,
                "anomaly_score": float (0-100 scale),
                "confidence": float (0-1),
                "raw_score": float
            }
        """
        if not self._is_trained:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "confidence": 0.0,
                "raw_score": 0.0,
            }

        features = self.extract_features(action)
        raw_score = self.iso_forest.decision_function(features)
        prediction = self.iso_forest.predict(features)

        # Convert raw score to 0-100 scale
        # Isolation Forest: more negative = more anomalous
        # Typical range: -0.5 (very anomalous) to 0.5 (normal)
        anomaly_score = max(0.0, min(100.0, (0.5 - raw_score) * 100.0))

        return {
            "is_anomaly": prediction == -1,
            "anomaly_score": round(anomaly_score, 2),
            "confidence": 0.85 if prediction == -1 else 0.95,
            "raw_score": round(float(raw_score), 4),
        }

    def generate_explanation(
        self,
        action: AgentAction,
        rule_violations: List[RuleViolation],
        memory_warnings: List[MemoryWarning],
        anomaly_result: Dict[str, Any],
        final_score: float,
    ) -> str:
        """
        Generate a human-readable explanation of the risk assessment.
        Uses template-based generation (no external API needed).
        """
        parts = []

        if rule_violations:
            rule_names = ", ".join(v.rule for v in rule_violations)
            parts.append(f"Policy violations detected: {rule_names}.")

        if memory_warnings:
            fraud_warnings = [w for w in memory_warnings if w.type == "fraud_pattern"]
            freq_warnings = [w for w in memory_warnings if w.type == "frequency_pattern"]
            if fraud_warnings:
                parts.append(f"Historical fraud pattern found for this customer.")
            if freq_warnings:
                parts.append(f"Unusual frequency of similar actions detected.")

        if anomaly_result.get("is_anomaly"):
            score = anomaly_result.get("anomaly_score", 0)
            parts.append(f"Statistical anomaly detected (score: {score:.0f}%).")

            # Add time-based context
            try:
                ts = datetime.fromisoformat(action.timestamp.replace("Z", "+00:00"))
                if ts.hour < 6 or ts.hour > 22:
                    parts.append(f"Action attempted at unusual hour ({ts.hour}:00).")
                if ts.weekday() >= 5:
                    parts.append("Action attempted on weekend.")
            except Exception:
                pass

        if not parts:
            return "Action passes all governance checks. No policy violations, historical warnings, or statistical anomalies detected. Safe to proceed."

        # Add risk level summary
        if final_score > 70:
            parts.append("Overall risk level: HIGH. Recommended action: BLOCK.")
        elif final_score > 40:
            parts.append("Overall risk level: MODERATE. Recommended action: ESCALATE for human review.")

        return " ".join(parts)

    def calculate_final_score(
        self,
        action: AgentAction,
        rule_violations: List[RuleViolation],
        memory_warnings: List[MemoryWarning],
        anomaly_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine all scoring pillars into a final risk score and recommendation.
        
        Returns:
            {
                "score": float (0-100),
                "recommendation": "APPROVE" | "BLOCK" | "ESCALATE",
                "explanation": str,
                "breakdown": { rule_score, memory_score, anomaly_score }
            }
        """
        # Individual pillar scores (0-100 each)
        rule_score = 100.0 if rule_violations else 0.0
        memory_score = 80.0 if memory_warnings else 0.0
        anomaly_score_val = anomaly_result.get("anomaly_score", 0.0)

        # Boost memory score if fraud detected
        fraud_count = sum(1 for w in memory_warnings if w.type == "fraud_pattern")
        if fraud_count > 0:
            memory_score = min(100.0, 80.0 + fraud_count * 10.0)

        # Weighted combination
        final_score = (
            settings.WEIGHT_RULES * rule_score
            + settings.WEIGHT_MEMORY * memory_score
            + settings.WEIGHT_ANOMALY * anomaly_score_val
        )
        final_score = min(100.0, max(0.0, final_score))

        # Determine recommendation
        if final_score > settings.RISK_BLOCK_THRESHOLD:
            recommendation = "BLOCK"
        elif final_score > settings.RISK_ESCALATE_THRESHOLD:
            recommendation = "ESCALATE"
        else:
            recommendation = "APPROVE"

        # Generate explanation
        explanation = self.generate_explanation(
            action, rule_violations, memory_warnings, anomaly_result, final_score
        )

        return {
            "score": round(final_score, 2),
            "recommendation": recommendation,
            "explanation": explanation,
            "breakdown": {
                "rule_score": round(rule_score, 2),
                "memory_score": round(memory_score, 2),
                "anomaly_score": round(anomaly_score_val, 2),
            },
        }
