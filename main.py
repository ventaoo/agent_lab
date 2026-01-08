from src.core.experiment import MultiAgentExperiment
from src.utils.config import Config
import json
import pandas as pd
import numpy as np
from src.utils.visualization import ExperimentVisualizer
from src.core.analysis import PostExperimentAnalysis
from src.core.contract_tests import ContractTestSuite
from src.core.security_tests import SecurityTestSuite

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def main():
    # Load configuration
    config = Config()
    
    # Initialize experiment
    experiment = MultiAgentExperiment(config)
    
    # Run experiment
    results = experiment.run_experiment(episodes=config.get('experiment.episodes', 1000))
    
    # Convert numpy types to native Python types
    results_converted = convert_numpy_types(results)
    
    # Save detailed results
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, ensure_ascii=False, indent=2)
    
    # Save metrics over time as CSV for easy analysis
    metrics_df = pd.DataFrame(results_converted['metrics_over_time'])
    metrics_df.to_csv('metrics_over_time.csv', index=False)
    
    # Save drift analysis as CSV
    drift_df = pd.DataFrame(results_converted['drift_analysis'])
    drift_df.to_csv('drift_analysis.csv', index=False)
    
    print("Results saved to:")
    print("- results.json (complete results)")
    print("- metrics_over_time.csv (metrics for each episode)")
    print("- drift_analysis.csv (drift metrics for each episode)")
    
    print("\nOverall Statistics:")
    stats = results_converted['overall_stats']
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Average Success Rate: {stats['avg_success_rate']:.3f}")
    print(f"Average KL Divergence: {stats['avg_kl_divergence']:.3f}")
    print(f"Average Embedding Shift: {stats['avg_embedding_shift']:.3f}")
    print(f"Average Response Similarity: {stats['avg_response_similarity']:.3f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualizer = ExperimentVisualizer()
    
    # Generate all plots
    visualizer.plot_all_metrics(save_path='all_metrics.png')
    visualizer.plot_success_rate_with_trend(window_size=50, save_path='success_rate_trend.png')
    visualizer.plot_drift_metrics_comparison(save_path='drift_comparison.png')
    visualizer.plot_correlation_heatmap(save_path='correlation_heatmap.png')
    visualizer.plot_drift_vs_performance(save_path='drift_vs_performance.png')
    visualizer.plot_episode_distribution(save_path='distributions.png')
    
    # Print summary statistics
    visualizer.print_summary()
    
    print("\nAll visualizations saved as PNG files!")
    
    # Part 2: Analysis after 1000 episodes
    print("\n" + "="*60)
    print("PART 2: ANALYSIS AFTER 1000 EPISODES")
    print("="*60)
    
    analysis = PostExperimentAnalysis(metrics_df, drift_df)
    
    # Run comprehensive analysis
    comprehensive_report = analysis.generate_comprehensive_report()
    
    print("Comprehensive Analysis Report:")
    print(comprehensive_report['summary'])
    
    # Print detailed analysis
    print("\nConvergence Analysis:")
    for metric, data in comprehensive_report['convergence_analysis'].items():
        print(f"  {metric}: Converged={data['converged']}, Mean Diff={data['mean_difference']:.4f}")
    
    print("\nStability Analysis:")
    for metric, data in comprehensive_report['stability_analysis'].items():
        print(f"  {metric}: Stability Score={data['stability_score']:.4f}")
    
    print("\nDrift Patterns:")
    for metric, data in comprehensive_report['drift_patterns'].items():
        print(f"  {metric}: Trend Slope={data['trend_slope']:.4f}, R²={data['trend_r_squared']:.4f}")
    
    print("\nDegradation Analysis:")
    for metric, data in comprehensive_report['degradation_analysis'].items():
        print(f"  {metric}: Degraded={data['degraded']}, Score={data['degradation_score']:.4f}")
    
    # Generate long-term trend plot
    analysis.plot_long_term_trends(save_path='long_term_trends.png')
    
    # Part 2: Contract Test Suite
    print("\n" + "="*60)
    print("CONTRACT TEST SUITE")
    print("="*60)
    
    # Need to recreate agent manager for testing
    from src.agents.agent_manager import AgentManager
    test_agent_manager = AgentManager(config)
    
    contract_tests = ContractTestSuite(test_agent_manager, metrics_df)
    test_results = contract_tests.run_all_contract_tests()
    
    print(contract_tests.generate_test_report())
    
    # Part 2: Security Tests (Additional)
    print("\n" + "="*60)
    print("SECURITY TEST SUITE - ATTACK SCENARIO SIMULATION")
    print("="*60)
    
    security_tests = SecurityTestSuite(test_agent_manager)
    security_audit = security_tests.run_security_audit()
    
    print("Security Audit Results:")
    print(f"Attack Success: {security_audit['security_audit_results']['attack_success']}")
    print(f"Attack Detected: {security_audit['security_audit_results']['attack_detected']}")
    print(f"Protection Status: {security_audit['protection_status']['status']}")
    
    print("\nRecommended Protection Mechanisms:")
    for mechanism in security_audit['security_audit_results']['protection_proposals']:
        print(f"  - {mechanism}")
    
    print("\nVulnerability Assessment:")
    for vuln, level in security_audit['vulnerability_assessment'].items():
        print(f"  {vuln}: {level}")
    
    print("\nSecurity tests completed!")

if __name__ == "__main__":
    main()