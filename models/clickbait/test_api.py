"""Test script for the Clickbait Detection API."""
import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start the server with: python run.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze(title, content=""):
    """Test the analyze endpoint."""
    print(f"\nTesting analyze endpoint...")
    print(f"Title: {title}")
    if content:
        print(f"Content: {content[:50]}...")
    
    try:
        data = {
            "title": title,
            "content": content
        }
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/analyze", json=data)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Label: {result['label']}")
            print(f"   Explanation: {result['explanation']}")
            print(f"   Response time: {elapsed_time:.2f}s")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Clickbait Detection API Test Suite")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n⚠️  Server is not running. Please start it first:")
        print("   python run.py")
        return
    
    # Test cases
    test_cases = [
        {
            "title": "You Won't Believe What Happened Next!",
            "content": "",
            "expected": "clickbait"
        },
        {
            "title": "Scientists Discover New Breakthrough in Medicine",
            "content": "Researchers at MIT have announced a new medical breakthrough that could revolutionize treatment.",
            "expected": "clean"
        },
        {
            "title": "This One Trick Will Change Your Life Forever!",
            "content": "",
            "expected": "clickbait"
        },
        {
            "title": "Local News: City Council Meeting Scheduled",
            "content": "The city council will meet next Tuesday to discuss the budget proposal.",
            "expected": "clean"
        },
        {
            "title": "SHOCKING: The Truth They Don't Want You to Know!",
            "content": "",
            "expected": "clickbait"
        }
    ]
    
    print("\n" + "=" * 60)
    print("Running Test Cases")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        result = test_analyze(test_case["title"], test_case["content"])
        if result:
            passed += 1
        else:
            failed += 1
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()


