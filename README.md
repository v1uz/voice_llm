# 🤖 AI Voice Agent - Autonomous Voice Assistant

Голосовой AI-агент с возможностью планирования и выполнения сложных задач. Объединяет голосовой интерфейс, автономное планирование, систему инструментов и память.

## ✨ Возможности

### 🎯 Автономный агент
- **Планирование задач** - разбивает сложные задачи на шаги
- **Система инструментов** - 12+ готовых инструментов
- **Память** - краткосрочная и долговременная память
- **Саморефлексия** - агент анализирует свои действия
- **Адаптивность** - корректирует план при ошибках

### 🗣️ Голосовой интерфейс
- **Распознавание речи** - Faster Whisper (локально)
- **Синтез речи** - Edge TTS (натуральные голоса)
- **VAD** - автоматическое определение речи
- **Прерывание** - можно остановить агента голосом

### 🛠️ Доступные инструменты

#### Веб
- `WebSearchTool` - поиск в Google
- `WebBrowserTool` - открытие сайтов
- `WebFetchTool` - загрузка контента страниц
- `YouTubeSearchTool` - поиск на YouTube

#### Файлы
- `FileReadTool` - чтение файлов
- `FileWriteTool` - создание/изменение файлов
- `FileListTool` - список файлов
- `FileOpenTool` - открытие файлов

#### Система
- `ShellCommandTool` - выполнение безопасных команд
- `PythonCodeTool` - выполнение Python кода
- `SystemInfoTool` - информация о системе
- `ApplicationLauncherTool` - запуск приложений

## 📦 Установка

### 1. Системные зависимости

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Windows:**
- Скачать PyAudio wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### 2. Ollama

Установите Ollama: https://ollama.ai

```bash
# Скачать модель
ollama pull llama3.2
```

### 3. Python зависимости

```bash
# Клонировать репозиторий
cd voice_llm

# Установить зависимости
pip install -r requirements.txt
```

## 🚀 Запуск

### Голосовой агент (рекомендуется)

```bash
python voice_agent.py
```

### Оригинальный голосовой ассистент

```bash
python voice.py
```

## 💡 Примеры использования

### Базовые команды

```
> [нажмите ENTER и скажите:]
"Search Google for Python tutorials"

> [или напечатайте:]
search for machine learning courses
```

### Работа с файлами

```
"Create a file called shopping.txt with milk, eggs, bread"
"Read the file shopping.txt"
"List all Python files in current directory"
```

### Веб-навигация

```
"Open youtube.com"
"Search YouTube for programming tutorials"
"Open google.com and search for weather"
```

### Выполнение команд

```
"Show system information"
"List files in current directory"
"Launch notepad"  # Windows
"Launch calculator"
```

### Планирование

```
/plan Create a Python script that reads data.csv and generates a report
```

Агент создаст план:
1. Read the file data.csv
2. Analyze the data
3. Generate report content
4. Write report to file

### Статус и рефлексия

```
/status   # Показать статус агента
/reflect  # Агент анализирует свои действия
```

## 🏗️ Архитектура

```
voice_llm/
├── agent/                    # Система агента
│   ├── core/
│   │   ├── agent.py         # Главный класс агента
│   │   └── planner.py       # Планировщик задач
│   ├── tools/
│   │   ├── base.py          # Базовый класс инструментов
│   │   ├── web_tools.py     # Веб-инструменты
│   │   ├── file_tools.py    # Файловые инструменты
│   │   └── system_tools.py  # Системные инструменты
│   └── memory/
│       └── agent_memory.py  # Система памяти
├── voice_agent.py           # Главный файл (агент + голос)
├── voice.py                 # Оригинальный ассистент
└── requirements.txt
```

## 🔧 Как это работает

### 1. Пользователь дает команду

```python
"Create a TODO list app in Python"
```

### 2. Агент создает план

```
📋 Plan: Create a TODO list app in Python
Tasks:
  1. ○ Design the TODO list structure
  2. ○ Write Python code for TODO app
  3. ○ Create file todo_app.py
  4. ○ Test the application
```

### 3. Агент выполняет каждый шаг

```
⏳ Executing: Design the TODO list structure
✓ Completed: Designed structure with add, remove, list functions

⏳ Executing: Write Python code for TODO app
🔧 Using tool: PythonCodeTool
✓ Completed: Generated Python code

⏳ Executing: Create file todo_app.py
🔧 Using tool: FileWriteTool
✓ Completed: File created successfully
```

### 4. Агент отчитывается

```
✓ Completed 4/4 tasks - Task accomplished!
```

## 🎮 Команды

### В процессе работы

- `ENTER` - начать говорить (голосовой ввод)
- Текст - написать команду напрямую
- `/plan <задача>` - показать план выполнения
- `/status` - статус агента
- `/reflect` - саморефлексия агента
- `quit` / `exit` - выход

## 🔒 Безопасность

### Встроенная защита

1. **Whitelist команд** - разрешены только безопасные команды
2. **Blacklist опасных операций** - блокировка rm, format, shutdown и т.д.
3. **Изолированное выполнение Python** - ограниченный namespace
4. **Валидация URL** - проверка корректности ссылок
5. **Timeout** - ограничение времени выполнения

### Дополнительные меры

Для production использования рекомендуется:
- Docker контейнер с ограниченными правами
- Песочница для выполнения кода
- Детальное логирование всех действий
- Rate limiting для API запросов

## 📊 Примеры сложных задач

### Исследование и отчет

```
"Research the top 3 Python frameworks and create a comparison report"
```

Агент:
1. Ищет информацию о фреймворках
2. Анализирует результаты
3. Создает файл с отчетом
4. Форматирует данные

### Автоматизация

```
"Find all .txt files, read them, and create a summary file"
```

Агент:
1. Сканирует директорию
2. Читает каждый .txt файл
3. Генерирует краткое содержание
4. Создает summary.txt

## 🛠️ Расширение

### Создание своего инструмента

```python
from agent.tools.base import Tool, ToolResult

class MyCustomTool(Tool):
    def get_description(self) -> str:
        return "Description of what your tool does"

    def get_schema(self) -> dict:
        return {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                }
            },
            "required": ["param1"]
        }

    def execute(self, param1: str) -> ToolResult:
        # Your logic here
        result = f"Processed: {param1}"

        return ToolResult(
            success=True,
            output=result
        )

# Регистрация
agent.tool_registry.register(MyCustomTool())
```

## 🐛 Отладка

### Включить подробные логи

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Проверить инструменты

```python
from agent import AIAgent

agent = AIAgent()
print(agent.get_status())
```

### Тестировать инструмент напрямую

```python
from agent.tools.web_tools import WebSearchTool

tool = WebSearchTool()
result = tool.run(query="Python tutorials")
print(result)
```

## 📈 Roadmap

- [ ] Поддержка русского языка
- [ ] Web UI (FastAPI + React)
- [ ] Поддержка плагинов
- [ ] RAG для работы с документами
- [ ] Интеграция с календарем/задачами
- [ ] Multi-agent collaboration
- [ ] Fine-tuning под специфические задачи

## 🤝 Вклад

Contributions are welcome! Пожалуйста:

1. Fork проект
2. Создайте feature branch
3. Commit изменения
4. Push в branch
5. Создайте Pull Request

## 📝 Лицензия

MIT License - свободно используйте в своих проектах

## 🙏 Благодарности

- OpenAI Whisper за отличное STT
- Ollama за локальные LLM
- Edge TTS за качественный синтез речи
- Anthropic за вдохновение архитектурой агентов

## 📧 Контакты

Вопросы? Предложения? Открывайте Issues!

---

**Важно:** Этот агент выполняет реальные действия на вашем компьютере. Всегда проверяйте, что он собирается делать, особенно с файлами и командами.

Enjoy! 🚀
