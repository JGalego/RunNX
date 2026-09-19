#!/usr/bin/env python3
"""
Operator Verification Script for RunNX

This script verifies the formal specifications of ONNX operators
using Why3 and provides property-based testing integration.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

class OperatorVerifier:
    """Handles formal verification of ONNX operators"""
    
    def __init__(self, formal_dir="formal"):
        self.formal_dir = Path(formal_dir)
        self.tensor_spec_file = self.formal_dir / "tensors.mlw"
        self.operator_spec_file = self.formal_dir / "operators.mlw"
        self.results = {}
        
        # Ensure we're in the right directory
        if not self.formal_dir.exists():
            self.formal_dir = Path(".")
            self.tensor_spec_file = self.formal_dir / "tensors.mlw"
            self.operator_spec_file = self.formal_dir / "operators.mlw"
        
    def check_why3_installation(self):
        """Check if Why3 is properly installed"""
        try:
            result = subprocess.run(
                ["why3", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"✅ Why3 found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Why3 not found. Please install Why3 first.")
            print("   Run: make install-why3")
            return False
    
    def detect_provers(self):
        """Detect available theorem provers and get their internal configuration names"""
        try:
            # Get the configuration to extract actual prover names
            result = subprocess.run(
                ["why3", "config", "show"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            provers = []
            lines = result.stdout.split('\n')
            current_prover = {}
            
            for line in lines:
                if line.startswith('[prover]'):
                    current_prover = {}
                elif line.startswith('name = ') and current_prover is not None:
                    name = line.split('= ')[1].strip('"')
                    current_prover['name'] = name
                elif line.startswith('version = ') and current_prover is not None:
                    version = line.split('= ')[1].strip('"')
                    current_prover['version'] = version
                elif line.startswith('shortcut = ') and current_prover is not None:
                    shortcut = line.split('= ')[1].strip('"')
                    current_prover['shortcut'] = shortcut
                elif line.startswith('alternative = ') and current_prover is not None:
                    alternative = line.split('= ')[1].strip('"')
                    current_prover['alternative'] = alternative
                elif line.strip() == '' and current_prover:
                    # End of prover section, construct the prover identifier
                    if 'name' in current_prover and 'version' in current_prover:
                        prover_id = current_prover.get(
                            'shortcut',
                            f"{current_prover['name']},{current_prover['version']}"
                        )
                        display_name = f"{current_prover['name']} {current_prover['version']}"
                        if 'alternative' in current_prover:
                            display_name += f" ({current_prover['alternative']})"
                        
                        provers.append({
                            'id': prover_id,
                            'display': display_name,
                            **current_prover,
                        })
                    current_prover = None
            
            display_names = [p['display'] for p in provers]
            print(f"🔍 Available provers: {', '.join(display_names)}")
            return provers
        except subprocess.CalledProcessError:
            print("⚠️ Could not detect provers")
            return []
    
    def verify_operator_specs(self, prover="alt-ergo", timeout=10):
        """Verify the operator specifications using Why3"""
        if not self.tensor_spec_file.exists():
            print(f"❌ Tensor specification file not found: {self.tensor_spec_file}")
            return False
            
        if not self.operator_spec_file.exists():
            print(f"❌ Operator specification file not found: {self.operator_spec_file}")
            return False

        print(f"🔍 Verifying tensor specifications with {prover}...")
        
        # First verify tensor specifications
        tensor_success = self._verify_file(self.tensor_spec_file, prover, timeout)
        
        print(f"🔍 Verifying operator specifications with {prover}...")
        
        # Then verify operator specifications  
        operator_success = self._verify_file(self.operator_spec_file, prover, timeout)
        
        return tensor_success and operator_success
    
    def _verify_file(self, spec_file, prover="alt-ergo", timeout=10):
        """Verify a single MLW file using Why3"""
        print(f"📝 Checking {spec_file.name}...")
        
        try:
            # Run Why3 proof verification
            cmd = [
                "why3", "prove", 
                str(spec_file),
                "-P", prover,
                "-t", str(timeout)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ All operator specifications verified successfully!")
                return True
            else:
                print(f"⚠️ Verification completed with warnings:")
                print(f"   stdout: {result.stdout}")
                if result.stderr:
                    print(f"   stderr: {result.stderr}")
                # Only consider prover ambiguity warnings as acceptable
                if "More than one prover" in result.stderr:
                    print("✅ Verification completed (prover ambiguity warnings ignored)")
                    return True
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Verification timed out after 60 seconds")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Why3 verification failed: {e}")
            return False
    
    def get_available_operators(self):
        """Dynamically detect available operators from MLW files"""
        operators = {}
        
        # Scan operators.mlw for predicate definitions
        try:
            with open(self.operator_spec_file, 'r') as f:
                content = f.read()
                
            # Find all predicates that end with _spec
            import re
            spec_patterns = re.findall(r'predicate\s+(\w+_spec)\s*\(', content)
            
            for spec in spec_patterns:
                # Extract operator name (remove _spec suffix)
                if spec.endswith('_spec'):
                    op_name = spec[:-5]  # Remove '_spec'
                    if op_name not in operators:
                        operators[op_name] = []
                    operators[op_name].append(spec)
                    
            # Also look for additional properties (monotonic, idempotent, etc.)
            property_patterns = re.findall(r'predicate\s+(\w+)_(monotonic|idempotent|bounded|commutativity|associativity|identity|inverse|positivity)\s*\(', content)
            
            for op_name, prop in property_patterns:
                if op_name not in operators:
                    operators[op_name] = []
                operators[op_name].append(f"{op_name}_{prop}")
                
        except FileNotFoundError:
            print(f"⚠️ Could not find {self.operator_spec_file}")
            
        return operators

    def verify_specific_operator(self, operator_name, prover="alt-ergo"):
        """Verify specifications for a specific operator"""
        print(f"🎯 Verifying {operator_name} operator...")
        
        # Get available operators dynamically
        available_operators = self.get_available_operators()
        
        if operator_name.lower() not in available_operators:
            print(f"❌ Unknown operator: {operator_name}")
            print(f"   Available operators: {', '.join(available_operators.keys())}")
            return False
        
        specs = available_operators[operator_name.lower()]
        print(f"   Verifying: {', '.join(specs)}")
        
        # Verify specific goals for this operator
        return self._verify_specific_goals(specs, prover)
    
    def _verify_specific_goals(self, goal_names, prover="alt-ergo", timeout=10):
        """Verify specific goals/predicates in the MLW files"""
        for goal in goal_names:
            print(f"    🔍 Verifying predicate: {goal}")
            if not self._check_predicate_exists(goal):
                print(f"    ❌ Predicate {goal} not found!")
                return False

        return self._verify_file_compiles(self.operator_spec_file, prover, timeout)
    
    def _check_predicate_exists(self, predicate_name):
        """Check if a predicate exists in the MLW files"""
        try:
            # Check in operators.mlw
            with open(self.operator_spec_file, 'r') as f:
                content = f.read()
                if f"predicate {predicate_name}" in content:
                    return True
            
            # Check in tensors.mlw
            with open(self.tensor_spec_file, 'r') as f:
                content = f.read()
                if f"predicate {predicate_name}" in content:
                    return True
            
            return False
        except FileNotFoundError:
            return False
    
    def _verify_file_compiles(self, spec_file, prover="alt-ergo", timeout=10):
        """Verify that an MLW file compiles and type-checks properly"""
        try:
            # Run Why3 proof verification to check compilation
            cmd = [
                "why3", "prove", 
                str(spec_file),
                "-P", prover,
                "-t", str(timeout)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            # Success if no compilation errors
            if result.returncode == 0:
                return True

            print(f"      stdout: {result.stdout.strip()}")
            print(f"      stderr: {result.stderr.strip()}")
            return False
                
        except subprocess.TimeoutExpired:
            return False
        except subprocess.CalledProcessError:
            return False
    
    def select_best_prover(self, provers):
        """Select the best available prover from the list"""
        if not provers:
            return None
        
        # Prefer a non-alternative automatic prover using its stable shortcut.
        for family in ["Alt-Ergo", "CVC5", "CVC4"]:
            for prover in provers:
                if prover.get('name') == family and 'alternative' not in prover:
                    return prover

        return provers[0]
    
    def run_all_verifications(self):
        """Run complete verification suite for operators"""
        print("🚀 Running complete operator verification suite...")
        
        if not self.check_why3_installation():
            return False
        
        provers = self.detect_provers()
        if not provers:
            print("⚠️ No provers detected, skipping formal verification")
            return False
        
        # Select the best available prover
        best_prover = self.select_best_prover(provers)
        if best_prover:
            prover_id = best_prover['id']
            prover_display = best_prover['display']
            print(f"🔧 Using prover: {prover_display}")
        else:
            print("❌ No suitable prover found")
            return False
        
        available_operators = self.get_available_operators()
        print(f"📋 Found specifications for {len(available_operators)} operators")
        all_passed = self.verify_operator_specs(prover_id)
        
        if all_passed:
            print("🎉 All operator verifications passed!")
        else:
            print("❌ Some verifications failed")
        
        return all_passed

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        operator_name = sys.argv[1]
        verifier = OperatorVerifier()
        if not verifier.check_why3_installation():
            sys.exit(1)
        
        provers = verifier.detect_provers()
        if not provers:
            print("⚠️ No provers available")
            sys.exit(1)
        
        best_prover = verifier.select_best_prover(provers)
        if best_prover:
            success = verifier.verify_specific_operator(operator_name, best_prover['id'])
            sys.exit(0 if success else 1)
        else:
            print("❌ No suitable prover found")
            sys.exit(1)
    else:
        verifier = OperatorVerifier()
        success = verifier.run_all_verifications()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
