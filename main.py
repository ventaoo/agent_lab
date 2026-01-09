import argparse
import json
from src.core.experiment_runner import ExperimentRunner
from tests.contract_tests import ContractTestSuite
from tests.security_tests import SecurityTestSuite
import unittest

def run_experiment():
    """Run the main experiment"""
    print("Starting Manager-Worker architecture experiment...")
    runner = ExperimentRunner()
    runner.run_experiment()
    print("Experiment completed successfully!")

def run_contract_tests():
    """Run contract tests"""
    print("Running contract tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(ContractTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_security_tests():
    """Run security tests"""
    print("Running security tests...")
    security_suite = SecurityTestSuite()
    
    # Run coordinated attack simulation
    attack_results = asyncio.run(security_suite.simulate_coordinated_attack())
    
    # Run global context awareness test
    context_results = security_suite.test_manager_global_context_awareness()
    
    print("Security tests completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Manager-Worker Architecture Experiment')
    parser.add_argument('--mode', choices=['experiment', 'contract_tests', 'security_tests', 'all'], 
                       default='experiment', help='Run mode')
    
    args = parser.parse_args()
    
    if args.mode in ['experiment', 'all']:
        run_experiment()
    
    if args.mode in ['contract_tests', 'all']:
        success = run_contract_tests()
        print(f"Contract tests {'PASSED' if success else 'FAILED'}")
    
    if args.mode in ['security_tests', 'all']:
        run_security_tests()

if __name__ == "__main__":
    import asyncio
    main()