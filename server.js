const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { execFile, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

// Убеждаемся, что директория uploads существует
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Глобальная переменная для хранения текущих процессов
let currentProcesses = new Set();
// Флаг остановки
let isStopping = false;

// Глобальная переменная для хранения процесса камеры
let cameraProcess = null;

// Глобальная переменная для хранения процесса IP-камеры
let ipCameraProcess = null;

// Настраиваем статическую раздачу файлов из папки с результатами
app.use('/result', express.static(path.join(__dirname, 'runs')));

// Настраиваем статическую раздачу файлов из папки uploads
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Настраиваем multer для загрузки файлов
const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const newFilename = `${file.fieldname}-${Date.now()}${ext}`;
    cb(null, newFilename);
  },
});
const upload = multer({ storage });

// Функция для создания директорий, если они не существуют
function ensureDirectoryExists(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

// Функция для записи логов в файл
function writeLogToFile(logPath, message) {
  // Создаем директорию для логов, если она не существует
  ensureDirectoryExists(path.dirname(logPath));
  
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(logPath, logMessage);
}

// Функция для поиска двух последних директорий predict
function findLatestPredictDirs() {
  const detectDir = path.join(__dirname, 'runs', 'detect');
  if (!fs.existsSync(detectDir)) {
    return ['predict', 'predict2']; // fallback
  }

  // Собираем только predictN, где N — число
  const dirs = fs.readdirSync(detectDir)
    .filter(dir => /^predict\d+$/.test(dir))
    .map(dir => {
      const num = parseInt(dir.replace('predict', ''));
      return { dir, num };
    })
    .sort((a, b) => b.num - a.num);

  // Берём только два последних: predictN и predictN-1
  if (dirs.length >= 2) {
    return [dirs[0].dir, dirs[1].dir];
  } else if (dirs.length === 1) {
    return [dirs[0].dir];
  } else {
    return [];
  }
}

// Функция для очистки старых директорий
function cleanupOldDirectories() {
  const detectDir = path.join(__dirname, 'runs', 'detect');
  if (!fs.existsSync(detectDir)) {
    return;
  }

  const dirs = fs.readdirSync(detectDir);
  dirs.forEach(dir => {
    if (dir.startsWith('predict') || dir.startsWith('predict_violence')) {
      const dirPath = path.join(detectDir, dir);
      if (fs.existsSync(dirPath)) {
        fs.rmSync(dirPath, { recursive: true, force: true });
        console.log(`Removed directory: ${dirPath}`);
      }
    }
  });
}

// Обработчик загрузки видеофайла
app.post('/upload-video', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  res.json({ filePath: req.file.path });
});

// Обработчик загрузки фотографии
app.post('/upload-photo', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  res.json({ filePath: req.file.path });
});

// Обработчик загрузки аудиофайла
app.post('/upload-audio', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  res.json({ filePath: req.file.path });
});

// Обработчик запуска анализа
app.get('/run-python', async (req, res) => {
  const filePath = req.query.filePath;
  const motionDetection = req.query.motionDetection === 'true';
  const nightMode = req.query.nightMode === 'true';
  const emotionDetection = req.query.emotionDetection === 'true';
  const quickSearch = req.query.quickSearch === 'true';

  if (!filePath) {
    return res.status(400).send('File path is required');
  }

  // Сбрасываем флаг остановки при новом запуске
  isStopping = false;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const sendSSE = (data) => {
    try {
      // Убеждаемся, что все строки в data правильно экранированы
      const sanitizedData = JSON.stringify(data)
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\t/g, '\\t');
      
      res.write(`data: ${sanitizedData}\n\n`);
    } catch (error) {
      console.error('Error sending SSE data:', error);
    }
  };

  try {
    // Очищаем старые директории перед запуском
    cleanupOldDirectories();

    // Запускаем модели последовательно
    const allModelResult = await runModel('all.pt', filePath, sendSSE, {
      motionDetection,
      nightMode,
      quickSearch
    });

    // Проверяем флаг остановки перед запуском второй модели
    if (isStopping) {
      sendSSE({
        status: 'complete',
        message: 'Обработка остановлена',
        resultPaths: [`/result/detect/predict/${path.basename(filePath)}`]
      });
      res.end();
      return;
    }

    // Если в режиме быстрого поиска нашли опасный объект, прекращаем обработку
    if (quickSearch && allModelResult.includes('WARNING: Dangerous objects detected')) {
      sendSSE({
        status: 'complete',
        message: 'Обработка остановлена: обнаружен опасный объект',
        resultPaths: [`/result/detect/predict/${path.basename(filePath)}`]
      });
      res.end();
      return;
    }

    // Запускаем модель violence.pt только если не было обнаружено опасных объектов
    const violenceModelResult = await runModel('violence.pt', filePath, sendSSE, {
      motionDetection,
      nightMode,
      quickSearch
    });

    // Проверяем флаг остановки перед отправкой результатов
    if (isStopping) {
      sendSSE({
        status: 'complete',
        message: 'Обработка остановлена',
        resultPaths: [
          `/result/detect/predict/${path.basename(filePath)}`,
          `/result/detect/predict_violence/${path.basename(filePath)}`
        ]
      });
      res.end();
      return;
    }

    // Запускаем распознавание эмоций, если включено и не было обнаружено опасных объектов
    let emotionResult = null;
    if (emotionDetection) {
      emotionResult = await runEmotionDetection(filePath, sendSSE);
    }

    // Получаем пути к сохраненным результатам
    const resultPaths = [
      `/result/detect/predict/${path.basename(filePath)}`,
      `/result/detect/predict_violence/${path.basename(filePath)}`
    ];

    // Добавляем путь к результатам эмоций, если они есть
    if (emotionResult) {
      resultPaths.push(`/result/detect/emotions/${path.basename(filePath)}`);
    }

    console.log('Sending result paths:', resultPaths);

    // Отправляем сообщение о завершении с путями к результатам
    sendSSE({
      status: 'complete',
      message: 'Обработка завершена',
      resultPaths: resultPaths,
      allModelResult,
      violenceModelResult,
      emotionResult
    });

    res.end();
  } catch (error) {
    console.error('Error running Python script:', error);
    sendSSE({ status: 'error', error: error.message });
    res.end();
  }
});

// Обработчик запуска анализа аудио
app.get('/run-audio-analysis', async (req, res) => {
  const filePath = req.query.filePath;

  if (!filePath) {
    return res.status(400).send('File path is required');
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const sendSSE = (data) => {
    // Убеждаемся, что все строки в data правильно экранированы
    const sanitizedData = JSON.stringify(data)
      .replace(/\n/g, '\\n')  // Экранируем переносы строк
      .replace(/\r/g, '\\r')  // Экранируем возвраты каретки
      .replace(/\t/g, '\\t'); // Экранируем табуляции
    
    res.write(`data: ${sanitizedData}\n\n`);
  };

  try {
    // Запускаем анализ аудио
    await runAudioAnalysis(filePath, sendSSE);
    
    // Отправляем только сообщение о завершении
    sendSSE({
      status: 'complete',
      message: 'Анализ завершен'
    });

    res.end();
  } catch (error) {
    console.error('Error running audio analysis:', error);
    sendSSE({ status: 'error', error: error.message });
    res.end();
  }
});

// Обработчик запуска анализа с камеры
app.get('/start-camera-analysis', async (req, res) => {
  const model = req.query.model;
  const cameraId = req.query.cameraId;

  console.log('Starting camera analysis with:', { model, cameraId });

  if (!model) {
    console.log('Missing model parameter');
    return res.status(400).send('Model is required');
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const sendSSE = (data) => {
    try {
      // Убеждаемся, что все строки в data правильно экранированы
      const sanitizedData = JSON.stringify(data)
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\t/g, '\\t');
      
      res.write(`data: ${sanitizedData}\n\n`);
    } catch (error) {
      console.error('Error sending SSE data:', error);
    }
  };

  try {
    // Останавливаем предыдущий процесс, если он существует
    if (cameraProcess) {
      console.log('Stopping previous camera process');
      cameraProcess.kill('SIGTERM');
      cameraProcess = null;
    }

    // Формируем полный путь к модели
    const modelPath = path.join(__dirname, 'yolo11', 'models', model);
    console.log('Model path:', modelPath);

    // Проверяем существование модели
    if (!fs.existsSync(modelPath)) {
      console.log('Model not found:', modelPath);
      sendSSE({ 
        status: 'error', 
        message: `Модель не найдена: ${model}` 
      });
      res.end();
      return;
    }

    // Запускаем новый процесс
    const scriptPath = path.join(__dirname, 'yolo11', 'camera_analysis.py');
    console.log('Starting camera analysis script:', scriptPath);
    
    cameraProcess = spawn('python3', [
      scriptPath,
      model
    ]);

    // Добавляем процесс в отслеживание
    addProcess(cameraProcess);

    let buffer = '';
    // Обработка вывода процесса
    cameraProcess.stdout.on('data', (data) => {
      try {
        buffer += data.toString();
        let newlineIndex;
        
        // Обрабатываем все полные JSON объекты в буфере
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          
          if (line) {
            try {
              const message = JSON.parse(line);
              console.log('Camera process output:', message);
              
              // Если это кадр, отправляем его отдельно
              if (message.status === 'frame' && message.image) {
                sendSSE({
                  status: 'frame',
                  image: message.image
                });
              } else {
                sendSSE(message);
              }
            } catch (e) {
              console.error('Error parsing camera output:', e, 'Raw line:', line);
            }
          }
        }
      } catch (error) {
        console.error('Error processing camera output:', error);
        sendSSE({ status: 'error', message: error.toString() });
      }
    });

    cameraProcess.stderr.on('data', (data) => {
      console.error('Camera process error:', data.toString());
      sendSSE({ status: 'error', message: data.toString() });
    });

    // Обработка завершения процесса
    cameraProcess.on('close', (code) => {
      console.log('Camera process closed with code:', code);
      if (code !== 0) {
        sendSSE({ status: 'error', message: `Camera process exited with code ${code}` });
      }
      cameraProcess = null;
    });

    // Обработка закрытия соединения
    req.on('close', () => {
      console.log('Client connection closed');
      if (cameraProcess) {
        cameraProcess.kill('SIGTERM');
        cameraProcess = null;
      }
    });

  } catch (error) {
    console.error('Error starting camera analysis:', error);
    sendSSE({ status: 'error', message: error.message });
    res.end();
  }
});

// Обработчик остановки анализа с камеры
app.get('/stop-camera-analysis', (req, res) => {
  if (cameraProcess) {
    cameraProcess.kill('SIGTERM');
    cameraProcess = null;
    res.json({ message: 'Camera analysis stopped' });
  } else {
    res.json({ message: 'No camera analysis running' });
  }
});

// Обработчик запуска анализа IP-камеры
app.get('/api/start-ip-camera', async (req, res) => {
    const model = req.query.model;
    const rtspUrl = req.query.rtspUrl;
    const motionDetection = req.query.motionDetection === 'true';
    const nightMode = req.query.nightMode === 'true';

    console.log('Starting IP camera analysis with:', { 
        model, 
        rtspUrl, 
        motionDetection, 
        nightMode,
        rawNightMode: req.query.nightMode 
    });

    if (!model || !rtspUrl) {
        console.log('Missing required parameters');
        return res.status(400).send('Model and RTSP URL are required');
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendSSE = (data) => {
        try {
            // Убеждаемся, что все строки в data правильно экранированы
            const sanitizedData = JSON.stringify(data)
                .replace(/\n/g, '\\n')
                .replace(/\r/g, '\\r')
                .replace(/\t/g, '\\t');
            
            // Отправляем данные в формате SSE
            res.write(`data: ${sanitizedData}\n\n`);
        } catch (error) {
            console.error('Error sending SSE data:', error);
        }
    };

    try {
        // Останавливаем предыдущий процесс, если он существует
        if (ipCameraProcess) {
            console.log('Stopping previous IP camera process');
            ipCameraProcess.kill('SIGTERM');
            ipCameraProcess = null;
        }

        // Формируем полные пути
        const scriptPath = path.join(__dirname, 'yolo11', 'ip_camera_analysis.py');
        const modelPath = path.join(__dirname, 'yolo11', 'models', model);
        
        console.log('Script path:', scriptPath);
        console.log('Model path:', modelPath);
        console.log('Night mode enabled:', nightMode);

        // Проверяем существование файлов
        if (!fs.existsSync(scriptPath)) {
            console.error('Script not found:', scriptPath);
            sendSSE({ status: 'error', message: 'Script not found' });
            res.end();
            return;
        }

        if (!fs.existsSync(modelPath)) {
            console.error('Model not found:', modelPath);
            sendSSE({ status: 'error', message: 'Model not found' });
            res.end();
            return;
        }

        // Запускаем новый процесс
        const args = [
            scriptPath,
            model,  // Передаем только имя модели, путь будет обработан в скрипте
            rtspUrl,
            motionDetection.toString(),
            nightMode.toString()
        ];
        
        console.log('Starting Python script with args:', args);
        
        ipCameraProcess = spawn('python3', args, {
            cwd: path.join(__dirname, 'yolo11'),  // Устанавливаем рабочую директорию
            env: {
                ...process.env,
                PYTHONPATH: path.join(__dirname, 'yolo11')  // Добавляем путь к Python модулям
            }
        });

        // Добавляем процесс в отслеживание
        addProcess(ipCameraProcess);

        let buffer = '';
        // Обработка вывода процесса
        ipCameraProcess.stdout.on('data', (data) => {
            try {
                buffer += data.toString();
                let newlineIndex;
                
                // Обрабатываем все полные JSON объекты в буфере
                while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                    const line = buffer.slice(0, newlineIndex).trim();
                    buffer = buffer.slice(newlineIndex + 1);
                    
                    if (line) {
                        try {
                            const message = JSON.parse(line);
                            console.log('IP camera process output:', message);
                            sendSSE(message);
                        } catch (e) {
                            console.error('Error parsing IP camera output:', e, 'Raw line:', line);
                        }
                    }
                }
            } catch (error) {
                console.error('Error processing IP camera output:', error);
                sendSSE({ status: 'error', message: error.toString() });
            }
        });

        ipCameraProcess.stderr.on('data', (data) => {
            const errorMessage = data.toString();
            console.error('IP camera process error:', errorMessage);
            sendSSE({ status: 'error', message: errorMessage });
        });

        // Обработка завершения процесса
        ipCameraProcess.on('close', (code) => {
            console.log('IP camera process closed with code:', code);
            if (code !== 0) {
                sendSSE({ status: 'error', message: `IP camera process exited with code ${code}` });
            }
            ipCameraProcess = null;
        });

        // Обработка закрытия соединения
        req.on('close', () => {
            console.log('Client connection closed');
            if (ipCameraProcess) {
                ipCameraProcess.kill('SIGTERM');
                ipCameraProcess = null;
            }
        });

    } catch (error) {
        console.error('Error starting IP camera analysis:', error);
        sendSSE({ status: 'error', message: error.message });
        res.end();
    }
});

// Обработчик остановки анализа IP-камеры
app.get('/api/stop-ip-camera', (req, res) => {
    if (ipCameraProcess) {
        ipCameraProcess.kill('SIGTERM');
        ipCameraProcess = null;
        res.json({ message: 'IP camera analysis stopped' });
    } else {
        res.json({ message: 'No IP camera analysis running' });
    }
});

// Функция для добавления процесса в отслеживание
function addProcess(process) {
  if (isStopping) {
    process.kill('SIGTERM');
    return;
  }
  currentProcesses.add(process);
  process.on('exit', () => {
    currentProcesses.delete(process);
  });
}

// Функция для остановки всех процессов
function stopAllProcesses() {
  isStopping = true;
  currentProcesses.forEach(process => {
    try {
      process.kill('SIGTERM');
    } catch (error) {
      console.error('Error stopping process:', error);
    }
  });
  currentProcesses.clear();
}

async function runModel(modelName, filePath, sendSSE, options = {}) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'yolo11', options.quickSearch ? 'quick_detect.py' : 'detect.py');
    const modelPath = path.join(__dirname, 'yolo11', 'models', modelName);
    const projectPath = path.join(__dirname, 'runs', 'detect');
    
    // Создаем директорию для результатов, если она не существует
    if (!fs.existsSync(projectPath)) {
      fs.mkdirSync(projectPath, { recursive: true });
    }

    // Используем разные имена директорий для разных моделей
    const predictDir = modelName === 'all.pt' ? 'predict' : 'predict_violence';

    const args = [
      scriptPath,
      '--weights', modelPath,
      '--source', filePath,
      '--conf', '0.40',
      '--save-txt',
      '--save',
      '--project', projectPath,
      '--name', predictDir,
      '--show'
    ];

    // Добавляем исключение классов только для модели all.pt
    if (modelName === 'all.pt') {
      args.push('--classes', 'antifa,bus,car,cat,celtic_cross,cigarette,cocaine,confederate-flag,destroy,dog,elephant,face,fire,glass-defect,gorilla,graffiti,gun,heroin,isis,knife,lion,marijuana,motorcycle,rocket,shrooms,smoke,squirrel,swastika,truck,wolfsangel,zebra');
    }

    if (options.motionDetection) {
      args.push('--motion-detection');
      sendSSE({ status: 'info', message: 'Датчик движения активирован' });
    }

    if (options.nightMode) {
      args.push('--night-mode');
      sendSSE({ status: 'info', message: 'Ночной режим активирован' });
    }

    console.log('Running command:', 'python3', ...args);

    const pythonProcess = spawn('python3', args);
    
    // Добавляем процесс в отслеживание
    addProcess(pythonProcess);

    let output = '';
    let detectedClasses = new Set();
    let dangerousObjectDetected = false;

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      console.log(chunk);
      
      // Проверяем на специальные сообщения режимов
      if (chunk.includes('Motion detected')) {
        sendSSE({ status: 'info', message: 'Motion detected: ' + chunk.trim() });
      } else if (chunk.includes('Night mode detected')) {
        sendSSE({ status: 'info', message: 'Night mode detected: ' + chunk.trim() });
      } else if (chunk.includes('WARNING: Motion detected in night scene!')) {
        dangerousObjectDetected = true;
        sendSSE({ 
          status: 'danger', 
          message: 'Обнаружено движение в ночной сцене!'
        });
        
        if (options.quickSearch) {
          console.log('Motion detected in night scene, stopping process...');
          pythonProcess.kill('SIGTERM');
          sendSSE({
            status: 'complete',
            message: 'Обработка остановлена: обнаружено движение в ночной сцене',
            resultPaths: [`/result/detect/${predictDir}/${path.basename(filePath)}`]
          });
          resolve(output);
          return;
        }
      } else if (chunk.includes('WARNING: Dangerous objects detected')) {
        dangerousObjectDetected = true;
        sendSSE({ 
          status: 'danger', 
          message: chunk.trim()
        });
        
        if (options.quickSearch) {
          console.log('Dangerous object detected in quick search mode, stopping process...');
          pythonProcess.kill('SIGTERM');
          sendSSE({
            status: 'complete',
            message: 'Обработка остановлена: обнаружен опасный объект',
            resultPaths: [`/result/detect/${predictDir}/${path.basename(filePath)}`]
          });
          resolve(output);
          return;
        }
      } else if (chunk.includes('Successfully saved frame to:')) {
        console.log('Frame saved:', chunk.trim());
        sendSSE({ status: 'info', message: 'Результаты сохранены' });
      } else if (chunk.includes('Exiting process due to')) {
        console.log('Process is exiting:', chunk.trim());
        pythonProcess.kill('SIGTERM');
      } else {
        // Ищем информацию о классах в выводе
        const classMatch = chunk.match(/detected (\d+) objects: (.+)/);
        if (classMatch) {
          const [, count, classes] = classMatch;
          const classList = classes.split(', ').map(c => c.trim());
          classList.forEach(c => detectedClasses.add(c));
          sendSSE({ 
            status: 'info', 
            message: `Обнаружено ${count} объектов: ${classList.join(', ')}`,
            classes: classList
          });
        } else {
          sendSSE({ status: 'info', message: chunk.trim() });
        }
      }
      
      const progressMatch = chunk.match(/video 1\/1 \(frame (\d+)\/(\d+)\)/);
      if (progressMatch) {
        const [, currentFrame, totalFrames] = progressMatch;
        const progress = Math.round((currentFrame / totalFrames) * 100);
        const detections = chunk.match(/detected (\d+) objects/);
        const detectedObjects = detections ? detections[1] : '0';
        sendSSE({
          status: 'progress',
          progress,
          currentFrame,
          totalFrames,
          detectedObjects,
          model: modelName
        });
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error: ${data}`);
      sendSSE({ status: 'error', message: data.toString() });
    });

    pythonProcess.on('close', (code) => {
      if (code === 0 || (code === null && dangerousObjectDetected)) {
        // Отправляем итоговый список обнаруженных классов
        if (detectedClasses.size > 0) {
          sendSSE({ 
            status: 'info', 
            message: `Итоговый список обнаруженных объектов: ${Array.from(detectedClasses).join(', ')}`,
            classes: Array.from(detectedClasses)
          });
        }
        resolve(output);
      } else {
        reject(new Error(`Python script exited with code ${code}`));
      }
    });
  });
}

// Добавляем функцию для запуска распознавания эмоций
async function runEmotionDetection(filePath, sendSSE) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'yolo11', 'emotion_detect.py');
    const projectPath = path.join(__dirname, 'runs', 'detect');
    
    // Создаем директорию для результатов, если она не существует
    if (!fs.existsSync(projectPath)) {
      fs.mkdirSync(projectPath, { recursive: true });
    }

    const args = [
      scriptPath,
      '--source', filePath,
      '--save',
      '--show',
      '--project', projectPath,
      '--name', 'emotions'
    ];

    console.log('Running emotion detection:', 'python3', ...args);

    const pythonProcess = spawn('python3', args);

    let output = '';
    let detectedEmotions = new Set();

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      console.log(chunk);
      
      // Обработка сообщений об эмоциях
      if (chunk.includes('Доминирующая эмоция:')) {
        const emotion = chunk.split('Доминирующая эмоция:')[1].trim();
        detectedEmotions.add(emotion);
        sendSSE({ 
          status: 'info', 
          message: `Доминирующая эмоция: ${emotion}`,
          emotion: emotion
        });
      } else {
        sendSSE({ status: 'info', message: chunk.trim() });
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error: ${data}`);
      sendSSE({ status: 'error', message: data.toString() });
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Emotion detection script exited with code ${code}`));
      }
    });
  });
}

// Функция для запуска анализа аудио
async function runAudioAnalysis(filePath, sendSSE) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'audio', 'Destructive_recognition.py');
    
    const args = [
      scriptPath,
      '--source', filePath
    ];

    console.log('Running audio analysis:', 'python3', ...args);

    const pythonProcess = spawn('python3', args);

    let output = '';

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      console.log(chunk);
      
      // Отправляем результаты анализа только один раз
      if (chunk.includes('Паралингвистический признак')) {
        sendSSE({ 
          status: 'info', 
          message: chunk.trim()
        });
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error: ${data}`);
      sendSSE({ status: 'error', message: data.toString() });
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Audio analysis script exited with code ${code}`));
      }
    });
  });
}

// Эндпоинт для остановки сервера
app.get('/stop-server', (req, res) => {
  // Останавливаем все процессы
  stopAllProcesses();

  // Отправляем ответ клиенту
  res.json({ message: 'Server stopping...' });

  // Перезапускаем сервер через 1 секунду
  setTimeout(() => {
    const newServer = spawn('node', ['server.js'], {
      detached: true,
      stdio: 'inherit'
    });
    newServer.unref();
    
    // Завершаем текущий процесс
    process.exit(0);
  }, 1000);
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Сервер запущен на порту ${PORT}`);
});
