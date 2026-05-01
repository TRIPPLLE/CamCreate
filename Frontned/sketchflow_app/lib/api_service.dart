import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  // Use 10.0.2.2 for Android Emulator, 127.0.0.1 for Web/Desktop
  // Assuming Web/Desktop for now
  static const String baseUrl = 'http://94.100.26.132:4567';

  static Future<Uint8List?> generateImage(Uint8List imageBytes) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/generate'));
      
      request.files.add(http.MultipartFile.fromBytes(
        'file', 
        imageBytes,
        filename: 'sketch.jpg',
        contentType: MediaType('image', 'jpeg'),
      ));

      var response = await request.send();

      if (response.statusCode == 200) {
        return await response.stream.toBytes();
      } else {
        print('Server error: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('API Error: $e');
      return null;
    }
  }
}
