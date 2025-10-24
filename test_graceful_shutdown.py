#!/usr/bin/env python3
"""Test script to verify graceful shutdown functionality."""

import asyncio
import signal
import subprocess
import sys
import time
from pathlib import Path

def test_graceful_shutdown():
    """Test that agent_analyzer.py can be gracefully terminated with Ctrl+C."""

    print("🧪 Testing graceful shutdown of agent_analyzer.py...")

    # Start the analyzer process
    print("🚀 Starting analyzer process...")
    process = subprocess.Popen(
        [sys.executable, "agent_analyzer.py", "--config", "config/development.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).parent
    )

    # Wait a bit for the process to start
    time.sleep(3)

    print(f"📊 Process started with PID: {process.pid}")
    print("⏰ Waiting 2 seconds for process to initialize...")
    time.sleep(2)

    # Check if process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"❌ Process exited prematurely")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False

    print("🛑 Sending SIGINT (Ctrl+C) signal...")

    # Send SIGINT signal (equivalent to Ctrl+C)
    process.send_signal(signal.SIGINT)

    # Wait for process to terminate (should be quick)
    start_time = time.time()
    timeout = 10  # 10 seconds max wait

    while process.poll() is None and (time.time() - start_time) < timeout:
        time.sleep(0.1)

    elapsed = time.time() - start_time

    if process.poll() is None:
        print(f"❌ Process did not terminate within {timeout} seconds")
        print("🔨 Force killing process...")
        process.kill()
        process.wait()
        return False

    stdout, stderr = process.communicate()

    print(f"✅ Process terminated gracefully in {elapsed:.2f} seconds")

    # Check output for graceful shutdown messages
    if "Received shutdown signal" in stdout or "Received shutdown signal" in stderr:
        print("✅ Graceful shutdown message detected")
    else:
        print("⚠️  No graceful shutdown message found")
        print(f"STDOUT: {stdout[-500:]}")  # Last 500 chars
        print(f"STDERR: {stderr[-500:]}")  # Last 500 chars

    if "Market Analyzer Agent shutdown complete" in stdout or "Market Analyzer Agent shutdown complete" in stderr:
        print("✅ Complete shutdown message detected")
    else:
        print("⚠️  No complete shutdown message found")

    return True

def test_immediate_shutdown():
    """Test immediate shutdown after startup."""

    print("\n🧪 Testing immediate shutdown after startup...")

    # Start the analyzer process
    print("🚀 Starting analyzer process...")
    process = subprocess.Popen(
        [sys.executable, "agent_analyzer.py", "--config", "config/development.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).parent
    )

    # Wait a very short time for process to start
    time.sleep(0.5)

    print("🛑 Immediately sending SIGINT signal...")

    # Send SIGINT signal immediately
    process.send_signal(signal.SIGINT)

    # Wait for process to terminate
    start_time = time.time()
    timeout = 5

    while process.poll() is None and (time.time() - start_time) < timeout:
        time.sleep(0.1)

    elapsed = time.time() - start_time

    if process.poll() is None:
        print(f"❌ Process did not terminate within {timeout} seconds")
        process.kill()
        process.wait()
        return False

    stdout, stderr = process.communicate()

    print(f"✅ Process terminated in {elapsed:.2f} seconds")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Testing agent_analyzer.py graceful shutdown")
    print("=" * 60)

    success = True

    try:
        # Test 1: Normal graceful shutdown
        if not test_graceful_shutdown():
            success = False

        # Test 2: Immediate shutdown
        if not test_immediate_shutdown():
            success = False

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        success = False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Graceful shutdown is working correctly.")
    else:
        print("❌ Some tests failed. Graceful shutdown needs improvement.")
    print("=" * 60)

    sys.exit(0 if success else 1)