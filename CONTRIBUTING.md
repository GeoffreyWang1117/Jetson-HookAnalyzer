# Contributing to HookAnalyzer

Thank you for your interest in contributing to HookAnalyzer!

## Development Setup

1. Fork and clone the repository
2. Build the project:
   ```bash
   ./scripts/build_local.sh debug
   ```
3. Run tests:
   ```bash
   cd build && ctest
   ```

## Code Style

- **C++**: Follow Google C++ Style Guide
- **CUDA**: Use descriptive kernel names, document block/grid sizes
- **Python**: Follow PEP 8

## Formatting

```bash
# C++ (clang-format)
clang-format -i core/**/*.{cpp,h}

# Python (black)
black api/
```

## Testing

- Add unit tests for new features
- Ensure all tests pass before submitting PR
- Include benchmark results for performance changes

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Submit PR with clear description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
