import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';

// Types
interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

interface ChatResponse {
    choices: Array<{
        message: {
            content: string;
        };
    }>;
}

interface ForgeConfig {
    apiUrl: string;
    apiKey: string;
    model: string;
    maxTokens: number;
    temperature: number;
}

// ForgeAI API Client
class ForgeAIClient {
    private client: AxiosInstance;
    private config: ForgeConfig;

    constructor() {
        this.config = this.loadConfig();
        this.client = this.createClient();
    }

    private loadConfig(): ForgeConfig {
        const config = vscode.workspace.getConfiguration('forgeai');
        return {
            apiUrl: config.get('apiUrl', 'http://localhost:8000'),
            apiKey: config.get('apiKey', ''),
            model: config.get('model', 'default'),
            maxTokens: config.get('maxTokens', 2048),
            temperature: config.get('temperature', 0.7)
        };
    }

    private createClient(): AxiosInstance {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json'
        };
        if (this.config.apiKey) {
            headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }
        return axios.create({
            baseURL: this.config.apiUrl,
            headers
        });
    }

    public refresh(): void {
        this.config = this.loadConfig();
        this.client = this.createClient();
    }

    public async chat(messages: ChatMessage[], systemPrompt?: string): Promise<string> {
        try {
            const payload: any = {
                messages,
                model: this.config.model,
                max_tokens: this.config.maxTokens,
                temperature: this.config.temperature
            };

            if (systemPrompt) {
                payload.messages = [
                    { role: 'system', content: systemPrompt },
                    ...messages
                ];
            }

            const response = await this.client.post<ChatResponse>(
                '/v1/chat/completions',
                payload
            );

            return response.data.choices[0]?.message?.content || '';
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`API Error: ${error.message}`);
            }
            throw error;
        }
    }

    public async complete(prompt: string, suffix?: string): Promise<string> {
        try {
            const response = await this.client.post('/v1/completions', {
                prompt,
                suffix,
                model: this.config.model,
                max_tokens: this.config.maxTokens,
                temperature: this.config.temperature
            });
            return response.data.choices[0]?.text || '';
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`API Error: ${error.message}`);
            }
            throw error;
        }
    }
}

// Chat Panel Provider
class ChatPanelProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'forgeai.chat';
    private _view?: vscode.WebviewView;
    private _messages: ChatMessage[] = [];
    private client: ForgeAIClient;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        client: ForgeAIClient
    ) {
        this.client = client;
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.type) {
                case 'sendMessage':
                    await this._handleUserMessage(message.content);
                    break;
                case 'clearChat':
                    this._messages = [];
                    this._updateChatView();
                    break;
            }
        });
    }

    private async _handleUserMessage(content: string) {
        this._messages.push({ role: 'user', content });
        this._updateChatView();

        try {
            const response = await this.client.chat(this._messages);
            this._messages.push({ role: 'assistant', content: response });
            this._updateChatView();
        } catch (error) {
            vscode.window.showErrorMessage(`ForgeAI Error: ${error}`);
        }
    }

    private _updateChatView() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateMessages',
                messages: this._messages
            });
        }
    }

    public addCodeContext(code: string, language: string) {
        const context = `\`\`\`${language}\n${code}\n\`\`\``;
        if (this._view) {
            this._view.webview.postMessage({
                type: 'addContext',
                content: context
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ForgeAI Chat</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            max-width: 85%;
        }
        .user-message {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            margin-left: auto;
        }
        .assistant-message {
            background: var(--vscode-editor-inactiveSelectionBackground);
            color: var(--vscode-editor-foreground);
        }
        .message pre {
            background: var(--vscode-textCodeBlock-background);
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .message code {
            font-family: var(--vscode-editor-font-family);
            font-size: var(--vscode-editor-font-size);
        }
        #input-container {
            padding: 10px;
            border-top: 1px solid var(--vscode-panel-border);
            display: flex;
            gap: 8px;
        }
        #message-input {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
            resize: vertical;
            min-height: 40px;
            max-height: 150px;
        }
        #send-button {
            padding: 8px 16px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #send-button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .loading {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--vscode-descriptionForeground);
        }
        .loading-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--vscode-progressBar-background);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div id="chat-container"></div>
    <div id="input-container">
        <textarea id="message-input" placeholder="Ask ForgeAI..." rows="1"></textarea>
        <button id="send-button">Send</button>
    </div>
    <script>
        const vscode = acquireVsCodeApi();
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        function formatMessage(content) {
            // Simple markdown to HTML
            return content
                .replace(/\`\`\`(\\w+)?\\n([\\s\\S]*?)\`\`\`/g, '<pre><code>$2</code></pre>')
                .replace(/\`([^\`]+)\`/g, '<code>$1</code>')
                .replace(/\\n/g, '<br>');
        }

        function renderMessages(messages) {
            chatContainer.innerHTML = messages.map(msg => {
                const className = msg.role === 'user' ? 'user-message' : 'assistant-message';
                return '<div class="message ' + className + '">' + formatMessage(msg.content) + '</div>';
            }).join('');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendMessage() {
            const content = messageInput.value.trim();
            if (!content) return;
            
            vscode.postMessage({ type: 'sendMessage', content });
            messageInput.value = '';
        }

        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        window.addEventListener('message', (event) => {
            const message = event.data;
            switch (message.type) {
                case 'updateMessages':
                    renderMessages(message.messages);
                    break;
                case 'addContext':
                    messageInput.value = message.content + '\\n\\n' + messageInput.value;
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}

// Inline Completion Provider
class ForgeCompletionProvider implements vscode.InlineCompletionItemProvider {
    private client: ForgeAIClient;
    private debounceTimer?: NodeJS.Timeout;

    constructor(client: ForgeAIClient) {
        this.client = client;
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | undefined> {
        const config = vscode.workspace.getConfiguration('forgeai');
        if (!config.get('inlineCompletions', true)) {
            return undefined;
        }

        // Get context around cursor
        const prefix = document.getText(new vscode.Range(
            new vscode.Position(Math.max(0, position.line - 50), 0),
            position
        ));
        const suffix = document.getText(new vscode.Range(
            position,
            new vscode.Position(Math.min(document.lineCount, position.line + 10), 0)
        ));

        try {
            const completion = await this.client.complete(prefix, suffix);
            if (completion && !token.isCancellationRequested) {
                return [new vscode.InlineCompletionItem(completion, new vscode.Range(position, position))];
            }
        } catch (error) {
            console.error('Completion error:', error);
        }

        return undefined;
    }
}

// Extension Activation
export function activate(context: vscode.ExtensionContext) {
    console.log('ForgeAI extension activated');

    const client = new ForgeAIClient();
    const chatProvider = new ChatPanelProvider(context.extensionUri, client);

    // Register chat view
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(ChatPanelProvider.viewType, chatProvider)
    );

    // Register inline completions
    const completionProvider = new ForgeCompletionProvider(client);
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            completionProvider
        )
    );

    // Configuration change listener
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('forgeai')) {
                client.refresh();
            }
        })
    );

    // Register commands
    const commands: Array<[string, (...args: any[]) => any]> = [
        ['forgeai.chat', () => {
            vscode.commands.executeCommand('workbench.view.extension.forgeai');
        }],
        
        ['forgeai.explainCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);
            if (!code) {
                vscode.window.showWarningMessage('Please select code to explain');
                return;
            }

            const language = editor.document.languageId;
            await executeWithProgress('Explaining code...', async () => {
                const explanation = await client.chat([
                    { role: 'user', content: `Explain this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\`` }
                ]);
                showResultPanel('Code Explanation', explanation);
            });
        }],

        ['forgeai.generateCode', async () => {
            const prompt = await vscode.window.showInputBox({
                prompt: 'Describe the code you want to generate',
                placeHolder: 'e.g., function to sort an array of objects by a property'
            });
            if (!prompt) return;

            const editor = vscode.window.activeTextEditor;
            const language = editor?.document.languageId || 'python';

            await executeWithProgress('Generating code...', async () => {
                const code = await client.chat([
                    { role: 'user', content: `Generate ${language} code for: ${prompt}\n\nOnly output the code, no explanations.` }
                ]);
                
                if (editor) {
                    // Insert at cursor
                    editor.edit(editBuilder => {
                        editBuilder.insert(editor.selection.active, code);
                    });
                } else {
                    // Open in new document
                    const doc = await vscode.workspace.openTextDocument({
                        content: code,
                        language
                    });
                    await vscode.window.showTextDocument(doc);
                }
            });
        }],

        ['forgeai.refactorCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);
            if (!code) {
                vscode.window.showWarningMessage('Please select code to refactor');
                return;
            }

            const instruction = await vscode.window.showInputBox({
                prompt: 'How should the code be refactored?',
                placeHolder: 'e.g., make it more readable, add error handling'
            });
            if (!instruction) return;

            const language = editor.document.languageId;
            await executeWithProgress('Refactoring code...', async () => {
                const refactored = await client.chat([
                    { role: 'user', content: `Refactor this ${language} code to ${instruction}. Only output the refactored code:\n\n\`\`\`${language}\n${code}\n\`\`\`` }
                ]);
                
                // Extract code from response
                const codeMatch = refactored.match(/```(?:\w+)?\n([\s\S]*?)```/);
                const cleanCode = codeMatch ? codeMatch[1] : refactored;
                
                editor.edit(editBuilder => {
                    editBuilder.replace(selection, cleanCode);
                });
            });
        }],

        ['forgeai.addComments', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);
            if (!code) {
                vscode.window.showWarningMessage('Please select code to comment');
                return;
            }

            const language = editor.document.languageId;
            await executeWithProgress('Adding comments...', async () => {
                const commented = await client.chat([
                    { role: 'user', content: `Add clear, concise comments to this ${language} code. Only output the commented code:\n\n\`\`\`${language}\n${code}\n\`\`\`` }
                ]);
                
                const codeMatch = commented.match(/```(?:\w+)?\n([\s\S]*?)```/);
                const cleanCode = codeMatch ? codeMatch[1] : commented;
                
                editor.edit(editBuilder => {
                    editBuilder.replace(selection, cleanCode);
                });
            });
        }],

        ['forgeai.generateTests', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);
            if (!code) {
                vscode.window.showWarningMessage('Please select code to generate tests for');
                return;
            }

            const language = editor.document.languageId;
            await executeWithProgress('Generating tests...', async () => {
                const tests = await client.chat([
                    { role: 'user', content: `Generate comprehensive unit tests for this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\`` }
                ]);
                
                // Open tests in new document
                const doc = await vscode.workspace.openTextDocument({
                    content: tests,
                    language
                });
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            });
        }],

        ['forgeai.findBugs', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);
            if (!code) {
                vscode.window.showWarningMessage('Please select code to analyze');
                return;
            }

            const language = editor.document.languageId;
            await executeWithProgress('Analyzing code...', async () => {
                const analysis = await client.chat([
                    { role: 'user', content: `Analyze this ${language} code for bugs, potential issues, and improvements:\n\n\`\`\`${language}\n${code}\n\`\`\`` }
                ]);
                showResultPanel('Bug Analysis', analysis);
            });
        }],

        ['forgeai.optimizeCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const code = editor.document.getText(selection);
            if (!code) {
                vscode.window.showWarningMessage('Please select code to optimize');
                return;
            }

            const language = editor.document.languageId;
            await executeWithProgress('Optimizing code...', async () => {
                const optimized = await client.chat([
                    { role: 'user', content: `Optimize this ${language} code for better performance. Explain the optimizations and provide the optimized code:\n\n\`\`\`${language}\n${code}\n\`\`\`` }
                ]);
                showResultPanel('Code Optimization', optimized);
            });
        }],

        ['forgeai.askAboutFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const question = await vscode.window.showInputBox({
                prompt: 'What would you like to know about this file?',
                placeHolder: 'e.g., what does this file do?'
            });
            if (!question) return;

            const content = editor.document.getText();
            const language = editor.document.languageId;
            const fileName = editor.document.fileName.split(/[\\/]/).pop();

            await executeWithProgress('Analyzing file...', async () => {
                const answer = await client.chat([
                    { role: 'user', content: `About this ${language} file (${fileName}):\n\n\`\`\`${language}\n${content}\n\`\`\`\n\nQuestion: ${question}` }
                ]);
                showResultPanel(`About ${fileName}`, answer);
            });
        }]
    ];

    for (const [id, handler] of commands) {
        context.subscriptions.push(vscode.commands.registerCommand(id, handler));
    }
}

// Helper: Show progress indicator
async function executeWithProgress<T>(title: string, task: () => Promise<T>): Promise<T | undefined> {
    return vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title,
            cancellable: false
        },
        async () => {
            try {
                return await task();
            } catch (error) {
                vscode.window.showErrorMessage(`ForgeAI Error: ${error}`);
                return undefined;
            }
        }
    );
}

// Helper: Show result in panel
function showResultPanel(title: string, content: string) {
    const panel = vscode.window.createWebviewPanel(
        'forgeaiResult',
        title,
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    // Simple markdown rendering
    const html = content
        .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');

    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            line-height: 1.6;
        }
        pre {
            background: var(--vscode-textCodeBlock-background);
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            font-family: var(--vscode-editor-font-family);
        }
    </style>
</head>
<body>${html}</body>
</html>`;
}

export function deactivate() {
    console.log('ForgeAI extension deactivated');
}
