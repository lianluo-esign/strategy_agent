# Trade Data Persistence Feature

## Overview

The Trade Data Persistence feature automatically serializes expired `trades_window` cache data to local disk as JSON files. This ensures that historical trade data older than 48 hours is preserved even after it expires from the Redis cache, providing data retention for analysis and compliance purposes.

## Architecture

### Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   New Trade     │───▶│   Redis Cache    │───▶│   Expired Data  │
│   Data (Min)    │    │  (trades_window) │    │   (>48h old)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                      │                    │
                                      ▼                    ▼
                              ┌──────────────────┐  ┌─────────────────┐
                              │   Analysis Use   │  │  File Storage   │
                              │   (Recent Data)  │  │   (JSON Files)  │
                              └──────────────────┘  └─────────────────┘
```

### Key Components

1. **Redis Client (`src/core/redis_client.py`)**
   - Handles expired data detection
   - Manages async file serialization
   - Implements concurrent file writing

2. **Data Collector (`src/agents/data_collector.py`)**
   - Configures storage directory
   - Triggers persistence during normal operation

3. **Configuration (`src/utils/config.py`)**
   - Configurable storage directory
   - Integration with existing config system

## Implementation Details

### Async File Serialization

The system uses `aiofiles` for non-blocking file operations and `asyncio.gather()` for concurrent file writing:

```python
async def _serialize_trade_data_to_files(self, expired_items: list[str]) -> None:
    """Serialize expired trade data to JSON files asynchronously."""
    try:
        # Create concurrent tasks for file writing
        tasks = [self._write_trade_data_file(item) for item in expired_items]

        # Execute all file writes concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        logger.error(f"Failed to serialize trade data to files: {e}")
```

### File Naming Convention

Files are named using the timestamp format: `trades_YYYYMMDD_HHMM.json`

- Example: `trades_20241023_1430.json` for trades from 2:30 PM on October 23, 2024
- One file per minute of trade data
- Ensures chronological ordering and easy file system navigation

### Data Structure

Each JSON file contains the complete minute trade data:

```json
{
  "timestamp": "2024-10-23T14:30:00.000000",
  "price_levels": {
    "60000.00": {
      "price_level": 60000.0,
      "buy_volume": 1.5,
      "sell_volume": 0.8,
      "total_volume": 2.3,
      "delta": 0.7,
      "trade_count": 15
    },
    "60001.00": {
      "price_level": 60001.0,
      "buy_volume": 0.9,
      "sell_volume": 1.2,
      "total_volume": 2.1,
      "delta": -0.3,
      "trade_count": 8
    }
  }
}
```

## Configuration

### Redis Configuration

Add the `storage_dir` parameter to your Redis configuration:

```yaml
redis:
  host: "localhost"
  port: 6379
  db: 0
  storage_dir: "storage"  # Directory for persistent trade files
```

### Directory Structure

The system automatically creates the storage directory if it doesn't exist:

```
project_root/
├── storage/
│   ├── trades_20241023_1430.json
│   ├── trades_20241023_1431.json
│   ├── trades_20241023_1432.json
│   └── ...
├── src/
├── config/
└── ...
```

## Performance Characteristics

### Memory Usage

- **Redis Memory**: Maintains 48-hour sliding window (2,880 minutes of data)
- **File System**: Persistent storage for all historical data
- **Concurrent Processing**: Multiple files written simultaneously without blocking main flow

### I/O Patterns

- **Non-blocking**: All file operations are async using `aiofiles`
- **Concurrent**: Multiple files written in parallel using `asyncio.gather()`
- **Error Resilient**: File write failures don't interrupt trade data collection

### Scalability

- **Horizontal**: Can handle thousands of minute files efficiently
- **Vertical**: Memory usage remains constant regardless of historical data volume
- **Performance**: File I/O is optimized to avoid blocking market data processing

## Error Handling

### Robust Error Recovery

The system implements comprehensive error handling at multiple levels:

1. **JSON Parsing Errors**: Invalid data is logged but doesn't crash the system
2. **File System Errors**: Disk full or permission issues are handled gracefully
3. **Redis Errors**: Connection issues don't prevent file operations
4. **Concurrent Failures**: Individual file write failures don't affect other files

```python
try:
    data = json.loads(data_str)
    # ... file operations ...
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON data for file serialization: {e}")
except OSError as e:
    logger.error(f"File system error writing trade data: {e}")
except Exception as e:
    logger.error(f"Failed to write trade data file: {e}")
```

## Monitoring and Logging

### Log Levels

- **DEBUG**: Individual file serialization operations
- **INFO**: Batch serialization operations and counts
- **WARNING**: JSON parsing issues and minor errors
- **ERROR**: File system errors and critical failures

### Key Metrics

Monitor these metrics to ensure healthy operation:

- Number of files serialized per hour
- File write success/failure rates
- Disk space usage in storage directory
- Redis memory usage consistency

## Integration Guide

### Adding to Existing Projects

1. **Update Dependencies**:
   ```bash
   pip install aiofiles>=23.2.0
   ```

2. **Update Configuration**:
   ```yaml
   redis:
     storage_dir: "path/to/storage"
   ```

3. **No Code Changes Required**: The feature integrates automatically with existing `DataCollectorAgent`

### Data Analysis Usage

Access historical data for analysis:

```python
import json
from pathlib import Path
from datetime import datetime

def load_historical_trades(date: datetime, minute: int):
    """Load historical trade data for a specific minute."""
    filename = f"trades_{date.strftime('%Y%m%d_%H%M')}.json"
    filepath = Path("storage") / filename

    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# Example usage
trades_2pm = load_historical_trades(datetime(2024, 10, 23, 14, 30), 30)
```

## Testing

### Unit Tests

The feature includes comprehensive unit tests covering:

- Normal file serialization operations
- Error handling scenarios
- Concurrent file writing
- Redis integration
- Configuration validation

Run tests:
```bash
pytest tests/unit/test_trade_data_persistence.py -v
```

### Integration Testing

Test the complete data pipeline:
```bash
pytest tests/integration/test_trade_persistence_integration.py -v
```

## Production Considerations

### Disk Space Management

- **Monitoring**: Monitor disk space usage in the storage directory
- **Cleanup Strategy**: Implement file cleanup for old data as needed
- **Archive Strategy**: Consider compressing or moving old files to cold storage

### Backup and Recovery

- **Regular Backups**: Include storage directory in backup procedures
- **File Integrity**: Consider implementing checksum verification
- **Disaster Recovery**: Ensure storage directory is replicated in production

### Performance Optimization

- **SSD Storage**: Use SSD storage for better I/O performance
- **File System**: Ensure adequate IOPS for concurrent file operations
- **Monitoring**: Monitor file system performance under load

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure the storage directory is writable
   - Check file system permissions

2. **Disk Full**
   - Monitor disk space usage
   - Implement cleanup procedures

3. **JSON Parsing Errors**
   - Check Redis data integrity
   - Monitor error logs for corruption

4. **Performance Issues**
   - Monitor file I/O wait times
   - Consider storage optimization

### Debug Mode

Enable debug logging for detailed operation tracing:

```python
import logging
logging.getLogger('src.core.redis_client').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **File Compression**: Compress old files to save disk space
2. **Batch Processing**: Optimize for large numbers of expired items
3. **Data Validation**: Add schema validation for JSON files
4. **Metrics Collection**: Add Prometheus metrics for monitoring
5. **File Rotation**: Implement automatic file rotation policies

### Extension Points

The architecture supports easy extension:

- Custom serialization formats (Parquet, Avro)
- Alternative storage backends (S3, cloud storage)
- Data transformation pipelines
- Real-time analytics integration

## Security Considerations

### File System Security

- Ensure proper file permissions on storage directory
- Consider file encryption for sensitive data
- Implement access controls for historical data

### Data Privacy

- Review data retention policies
- Consider data anonymization for long-term storage
- Ensure compliance with relevant regulations

## Version History

- **v1.0.0**: Initial implementation with async file serialization
- **v1.0.1**: Added comprehensive error handling and improved logging
- **v1.1.0**: Enhanced performance with concurrent file writing
- **v1.2.0**: Added comprehensive test suite and quality improvements