import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:signature/signature.dart';
import 'package:image_picker/image_picker.dart';
import 'api_service.dart';

void main() {
  runApp(const SketchFlowApp());
}

class SketchFlowApp extends StatelessWidget {
  const SketchFlowApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SketchFlowGAN',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blueAccent,
          brightness: Brightness.dark,
        ),
        fontFamily: 'Roboto', // Default fallback, modern look
      ),
      home: const SketchFlowHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SketchFlowHome extends StatefulWidget {
  const SketchFlowHome({super.key});

  @override
  State<SketchFlowHome> createState() => _SketchFlowHomeState();
}

class _SketchFlowHomeState extends State<SketchFlowHome> {
  final SignatureController _signatureController = SignatureController(
    penStrokeWidth: 4,
    penColor: Colors.black,
    exportBackgroundColor: Colors.white,
  );

  Uint8List? _generatedImage;
  Uint8List? _uploadedImage;
  bool _isLoading = false;

  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      final bytes = await image.readAsBytes();
      setState(() {
        _uploadedImage = bytes;
        _signatureController.clear();
      });
    }
  }

  Future<void> _generatePhoto() async {
    setState(() {
      _isLoading = true;
    });

    Uint8List? inputBytes;

    if (_uploadedImage != null) {
      inputBytes = _uploadedImage;
    } else if (_signatureController.isNotEmpty) {
      inputBytes = await _signatureController.toPngBytes();
    } else {
      // Empty canvas
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please draw or upload a sketch first!')),
      );
      setState(() {
        _isLoading = false;
      });
      return;
    }

    if (inputBytes != null) {
      final resultBytes = await ApiService.generateImage(inputBytes);
      if (resultBytes != null) {
        setState(() {
          _generatedImage = resultBytes;
        });
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Failed to generate image. Is the backend running?')),
          );
        }
      }
    }

    setState(() {
      _isLoading = false;
    });
  }

  void _clear() {
    _signatureController.clear();
    setState(() {
      _uploadedImage = null;
      _generatedImage = null;
    });
  }

  @override
  void dispose() {
    _signatureController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Determine layout based on screen width
    final isDesktop = MediaQuery.of(context).size.width > 800;

    return Scaffold(
      appBar: AppBar(
        title: const Text('🎨 SketchFlowGAN', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.transparent,
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            tooltip: 'Clear All',
            onPressed: _clear,
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Theme.of(context).colorScheme.background,
              Theme.of(context).colorScheme.primaryContainer.withOpacity(0.2),
            ],
          ),
        ),
        child: isDesktop ? _buildDesktopLayout() : _buildMobileLayout(),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isLoading ? null : _generatePhoto,
        icon: _isLoading 
            ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
            : const Icon(Icons.auto_awesome),
        label: const Text('Generate Photo'),
      ),
    );
  }

  Widget _buildDesktopLayout() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: _buildInputSection(),
          ),
          const SizedBox(width: 32),
          Expanded(
            child: _buildOutputSection(),
          ),
        ],
      ),
    );
  }

  Widget _buildMobileLayout() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          SizedBox(height: 400, child: _buildInputSection()),
          const SizedBox(height: 24),
          SizedBox(height: 400, child: _buildOutputSection()),
          const SizedBox(height: 80), // Space for FAB
        ],
      ),
    );
  }

  Widget _buildInputSection() {
    return Card(
      elevation: 8,
      shadowColor: Colors.black45,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Your Sketch', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                TextButton.icon(
                  onPressed: _pickImage,
                  icon: const Icon(Icons.upload_file),
                  label: const Text('Upload'),
                )
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: _uploadedImage != null
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.memory(_uploadedImage!, fit: BoxFit.contain),
                          Positioned(
                            top: 8, right: 8,
                            child: IconButton(
                              icon: const Icon(Icons.close, color: Colors.white),
                              style: IconButton.styleFrom(backgroundColor: Colors.black54),
                              onPressed: () => setState(() => _uploadedImage = null),
                            ),
                          )
                        ],
                      )
                    : Signature(
                        controller: _signatureController,
                        backgroundColor: Colors.white,
                      ),
              ),
            ),
            if (_uploadedImage == null)
              Padding(
                padding: const EdgeInsets.only(top: 12.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.undo),
                      onPressed: () => _signatureController.undo(),
                    ),
                    IconButton(
                      icon: const Icon(Icons.redo),
                      onPressed: () => _signatureController.redo(),
                    ),
                  ],
                ),
              )
          ],
        ),
      ),
    );
  }

  Widget _buildOutputSection() {
    return Card(
      elevation: 8,
      shadowColor: Colors.black45,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Generated Output', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceVariant,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: _generatedImage != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.memory(
                          _generatedImage!,
                          fit: BoxFit.contain,
                        ),
                      )
                    : const Center(
                        child: Text(
                          'Your generated photo will appear here',
                          style: TextStyle(color: Colors.grey),
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
