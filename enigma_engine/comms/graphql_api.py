"""
GraphQL API for Enigma AI Engine

Alternative GraphQL interface providing type-safe queries and mutations.

FILE: enigma_engine/comms/graphql_api.py
TYPE: API
MAIN CLASSES: GraphQLAPI, ForgeSchema

================================================================================
GRAPHQL SCHEMA DOCUMENTATION
================================================================================

## Types

### Message
    - id: String - Unique message identifier
    - role: String - "user" or "assistant"
    - content: String - Message text
    - timestamp: Float - Unix timestamp
    - metadata: String - JSON metadata

### Conversation
    - id: String - Conversation identifier
    - title: String - Conversation title
    - messages: [Message] - List of messages
    - createdAt: Float - Creation timestamp
    - updatedAt: Float - Last update timestamp

### ModelInfo
    - name: String - Model name
    - size: String - Model size category
    - loaded: Boolean - Whether model is currently loaded
    - parameters: Int - Number of parameters
    - contextLength: Int - Maximum context length

### GenerationResult
    - text: String - Generated text
    - tokensGenerated: Int - Number of tokens generated
    - generationTime: Float - Time taken in seconds
    - finishReason: String - Why generation stopped

### Module
    - name: String - Module name
    - category: String - Module category
    - status: String - "loaded", "unloaded", "error"
    - description: String - Module description
    - dependencies: [String] - Required dependencies

### Tool
    - name: String - Tool name
    - description: String - Tool description
    - parameters: String - JSON parameter schema

### SystemStatus
    - uptime: Float - System uptime in seconds
    - memoryUsedMB: Float - Memory usage
    - gpuMemoryUsedMB: Float - GPU memory usage
    - activeModules: [String] - List of loaded modules
    - modelLoaded: Boolean - Whether a model is loaded

## Queries

    conversations: [Conversation] - List all conversations
    conversation(id: String!): Conversation - Get specific conversation
    models: [ModelInfo] - List available models
    modules: [Module] - List all modules
    tools: [Tool] - List available tools
    systemStatus: SystemStatus - Get system status

## Mutations

    generate(prompt: String!, options: GenerationOptionsInput): GenerationResult
    sendMessage(conversationId: String, message: MessageInput!): Message
    createConversation(title: String): Conversation
    deleteConversation(id: String!): Boolean
    loadModule(name: String!): Module
    unloadModule(name: String!): Boolean

## Input Types

### MessageInput
    - role: String! - "user" or "assistant"
    - content: String! - Message text

### GenerationOptionsInput
    - maxTokens: Int
    - temperature: Float
    - topP: Float
    - topK: Int
    - stopSequences: [String]

## Example Queries

```graphql
# Generate text
mutation {
  generate(prompt: "Hello, AI!", options: {maxTokens: 100, temperature: 0.8}) {
    text
    tokensGenerated
    generationTime
  }
}

# List conversations
query {
  conversations {
    id
    title
    messages {
      role
      content
    }
  }
}

# Get system status
query {
  systemStatus {
    uptime
    memoryUsedMB
    modelLoaded
    activeModules
  }
}
```
================================================================================
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from graphql import (
        GraphQLArgument,
        GraphQLBoolean,
        GraphQLEnumType,
        GraphQLEnumValue,
        GraphQLField,
        GraphQLFloat,
        GraphQLInputField,
        GraphQLInputObjectType,
        GraphQLInt,
        GraphQLList,
        GraphQLNonNull,
        GraphQLObjectType,
        GraphQLSchema,
        GraphQLString,
        graphql,
        graphql_sync,
    )
    HAS_GRAPHQL = True
except ImportError:
    HAS_GRAPHQL = False

try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GraphiQL HTML interface (defined before if block for module-level access)
GRAPHIQL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Enigma AI Engine GraphQL</title>
    <link href="https://unpkg.com/graphiql/graphiql.min.css" rel="stylesheet" />
</head>
<body style="margin: 0;">
    <div id="graphiql" style="height: 100vh;"></div>
    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/graphiql/graphiql.min.js"></script>
    <script>
        const fetcher = GraphiQL.createFetcher({ url: '/graphql' });
        ReactDOM.render(
            React.createElement(GraphiQL, { fetcher: fetcher }),
            document.getElementById('graphiql'),
        );
    </script>
</body>
</html>
"""


if HAS_GRAPHQL:
    
    # GraphQL Types
    MessageType = GraphQLObjectType(
        "Message",
        lambda: {
            "id": GraphQLField(GraphQLString),
            "role": GraphQLField(GraphQLString),
            "content": GraphQLField(GraphQLString),
            "timestamp": GraphQLField(GraphQLFloat),
            "metadata": GraphQLField(GraphQLString)
        }
    )
    
    ConversationType = GraphQLObjectType(
        "Conversation",
        lambda: {
            "id": GraphQLField(GraphQLString),
            "title": GraphQLField(GraphQLString),
            "messages": GraphQLField(GraphQLList(MessageType)),
            "createdAt": GraphQLField(GraphQLFloat),
            "updatedAt": GraphQLField(GraphQLFloat)
        }
    )
    
    ModelInfoType = GraphQLObjectType(
        "ModelInfo",
        lambda: {
            "name": GraphQLField(GraphQLString),
            "size": GraphQLField(GraphQLString),
            "loaded": GraphQLField(GraphQLBoolean),
            "parameters": GraphQLField(GraphQLInt),
            "contextLength": GraphQLField(GraphQLInt)
        }
    )
    
    GenerationResultType = GraphQLObjectType(
        "GenerationResult",
        lambda: {
            "text": GraphQLField(GraphQLString),
            "tokensGenerated": GraphQLField(GraphQLInt),
            "generationTime": GraphQLField(GraphQLFloat),
            "finishReason": GraphQLField(GraphQLString)
        }
    )
    
    ModuleType = GraphQLObjectType(
        "Module",
        lambda: {
            "name": GraphQLField(GraphQLString),
            "category": GraphQLField(GraphQLString),
            "status": GraphQLField(GraphQLString),
            "description": GraphQLField(GraphQLString),
            "dependencies": GraphQLField(GraphQLList(GraphQLString))
        }
    )
    
    ToolType = GraphQLObjectType(
        "Tool",
        lambda: {
            "name": GraphQLField(GraphQLString),
            "description": GraphQLField(GraphQLString),
            "parameters": GraphQLField(GraphQLString)
        }
    )
    
    SystemStatusType = GraphQLObjectType(
        "SystemStatus",
        lambda: {
            "uptime": GraphQLField(GraphQLFloat),
            "memoryUsedMB": GraphQLField(GraphQLFloat),
            "gpuMemoryUsedMB": GraphQLField(GraphQLFloat),
            "activeModules": GraphQLField(GraphQLList(GraphQLString)),
            "modelLoaded": GraphQLField(GraphQLBoolean)
        }
    )
    
    # Input types
    MessageInput = GraphQLInputObjectType(
        "MessageInput",
        {
            "role": GraphQLInputField(GraphQLNonNull(GraphQLString)),
            "content": GraphQLInputField(GraphQLNonNull(GraphQLString))
        }
    )
    
    GenerationOptionsInput = GraphQLInputObjectType(
        "GenerationOptionsInput",
        {
            "maxTokens": GraphQLInputField(GraphQLInt),
            "temperature": GraphQLInputField(GraphQLFloat),
            "topP": GraphQLInputField(GraphQLFloat),
            "topK": GraphQLInputField(GraphQLInt),
            "stopSequences": GraphQLInputField(GraphQLList(GraphQLString))
        }
    )
    
    
    class GraphQLResolvers:
        """GraphQL resolvers for Enigma AI Engine."""
        
        def __init__(self):
            self._conversations: Dict[str, Dict] = {}
            self._start_time = time.time()
        
        # Query resolvers
        def resolve_conversations(self, info, limit=10, offset=0):
            """Get conversations."""
            convs = list(self._conversations.values())
            return convs[offset:offset + limit]
        
        def resolve_conversation(self, info, id):
            """Get single conversation."""
            return self._conversations.get(id)
        
        def resolve_models(self, info):
            """Get available models."""
            try:
                from enigma_engine.modules import ModuleManager
                manager = ModuleManager()
                
                models = []
                # Get model info
                model_mod = manager.get_module("model")
                if model_mod:
                    models.append({
                        "name": "forge-small",
                        "size": "small",
                        "loaded": True,
                        "parameters": 27000000,
                        "contextLength": 2048
                    })
                
                return models
            except (ImportError, AttributeError, Exception) as e:
                logger.debug(f"Failed to resolve models: {e}")
                return []
        
        def resolve_modules(self, info, category=None):
            """Get modules."""
            try:
                from enigma_engine.modules import ModuleManager
                manager = ModuleManager()
                
                modules = []
                for name, mod in manager._modules.items():
                    info_obj = manager.get_module_info(name)
                    if info_obj:
                        if category and info_obj.category.value != category:
                            continue
                        modules.append({
                            "name": name,
                            "category": info_obj.category.value,
                            "status": "loaded" if manager.is_loaded(name) else "available",
                            "description": info_obj.description,
                            "dependencies": list(info_obj.dependencies)
                        })
                
                return modules
            except (ImportError, AttributeError, Exception) as e:
                logger.debug(f"Failed to resolve modules: {e}")
                return []
        
        def resolve_tools(self, info):
            """Get available tools."""
            try:
                from enigma_engine.tools import get_all_tools
                tools = get_all_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": json.dumps(t.parameters)
                    }
                    for t in tools
                ]
            except (ImportError, AttributeError, Exception) as e:
                logger.debug(f"Failed to resolve tools: {e}")
                return []
        
        def resolve_system_status(self, info):
            """Get system status."""
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
            except (ImportError, Exception):
                memory_mb = 0
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                else:
                    gpu_memory = 0
            except (ImportError, RuntimeError):
                gpu_memory = 0
            
            return {
                "uptime": time.time() - self._start_time,
                "memoryUsedMB": memory_mb,
                "gpuMemoryUsedMB": gpu_memory,
                "activeModules": [],
                "modelLoaded": True
            }
        
        # Mutation resolvers
        def resolve_generate(self, info, prompt, options=None):
            """Generate text."""
            try:
                from enigma_engine.core.inference import EnigmaEngine
                
                engine = EnigmaEngine()
                
                max_tokens = 256
                temperature = 0.7
                
                if options:
                    max_tokens = options.get("maxTokens", 256)
                    temperature = options.get("temperature", 0.7)
                
                start = time.time()
                result = engine.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                gen_time = time.time() - start
                
                return {
                    "text": result.text if hasattr(result, "text") else str(result),
                    "tokensGenerated": len(result.text.split()) if hasattr(result, "text") else 0,
                    "generationTime": gen_time,
                    "finishReason": "complete"
                }
            except Exception as e:
                return {
                    "text": f"Error: {e}",
                    "tokensGenerated": 0,
                    "generationTime": 0,
                    "finishReason": "error"
                }
        
        def resolve_create_conversation(self, info, title):
            """Create conversation."""
            import uuid
            conv_id = str(uuid.uuid4())
            
            conv = {
                "id": conv_id,
                "title": title,
                "messages": [],
                "createdAt": time.time(),
                "updatedAt": time.time()
            }
            
            self._conversations[conv_id] = conv
            return conv
        
        def resolve_send_message(self, info, conversationId, message):
            """Send message to conversation."""
            import uuid
            
            if conversationId not in self._conversations:
                return None
            
            conv = self._conversations[conversationId]
            
            # Add user message
            user_msg = {
                "id": str(uuid.uuid4()),
                "role": message["role"],
                "content": message["content"],
                "timestamp": time.time(),
                "metadata": "{}"
            }
            conv["messages"].append(user_msg)
            
            # Generate response if user message
            if message["role"] == "user":
                try:
                    from enigma_engine.core.inference import EnigmaEngine
                    engine = EnigmaEngine()
                    
                    response = engine.generate(message["content"])
                    
                    assistant_msg = {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": response.text if hasattr(response, "text") else str(response),
                        "timestamp": time.time(),
                        "metadata": "{}"
                    }
                    conv["messages"].append(assistant_msg)
                except (ImportError, Exception) as e:
                    logger.debug(f"Failed to generate response: {e}")
            
            conv["updatedAt"] = time.time()
            return conv
        
        def resolve_load_module(self, info, name):
            """Load module."""
            try:
                from enigma_engine.modules import ModuleManager
                manager = ModuleManager()
                manager.load(name)
                
                info_obj = manager.get_module_info(name)
                return {
                    "name": name,
                    "category": info_obj.category.value if info_obj else "unknown",
                    "status": "loaded",
                    "description": info_obj.description if info_obj else "",
                    "dependencies": []
                }
            except Exception as e:
                return None
        
        def resolve_unload_module(self, info, name):
            """Unload module."""
            try:
                from enigma_engine.modules import ModuleManager
                manager = ModuleManager()
                manager.unload(name)
                return True
            except (ImportError, Exception) as e:
                logger.debug(f"Failed to unload module {name}: {e}")
                return False
        
        def resolve_execute_tool(self, info, name, arguments):
            """Execute tool."""
            try:
                from enigma_engine.tools import ToolExecutor
                executor = ToolExecutor()
                
                args = json.loads(arguments) if arguments else {}
                result = executor.execute(name, args)
                
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})
    
    
    def create_schema(resolvers: GraphQLResolvers) -> GraphQLSchema:
        """Create GraphQL schema."""
        
        # Query type
        query_type = GraphQLObjectType(
            "Query",
            {
                "conversations": GraphQLField(
                    GraphQLList(ConversationType),
                    args={
                        "limit": GraphQLArgument(GraphQLInt),
                        "offset": GraphQLArgument(GraphQLInt)
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_conversations(info, **args)
                ),
                "conversation": GraphQLField(
                    ConversationType,
                    args={
                        "id": GraphQLArgument(GraphQLNonNull(GraphQLString))
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_conversation(info, **args)
                ),
                "models": GraphQLField(
                    GraphQLList(ModelInfoType),
                    resolve=lambda obj, info: resolvers.resolve_models(info)
                ),
                "modules": GraphQLField(
                    GraphQLList(ModuleType),
                    args={
                        "category": GraphQLArgument(GraphQLString)
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_modules(info, **args)
                ),
                "tools": GraphQLField(
                    GraphQLList(ToolType),
                    resolve=lambda obj, info: resolvers.resolve_tools(info)
                ),
                "systemStatus": GraphQLField(
                    SystemStatusType,
                    resolve=lambda obj, info: resolvers.resolve_system_status(info)
                )
            }
        )
        
        # Mutation type
        mutation_type = GraphQLObjectType(
            "Mutation",
            {
                "generate": GraphQLField(
                    GenerationResultType,
                    args={
                        "prompt": GraphQLArgument(GraphQLNonNull(GraphQLString)),
                        "options": GraphQLArgument(GenerationOptionsInput)
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_generate(info, **args)
                ),
                "createConversation": GraphQLField(
                    ConversationType,
                    args={
                        "title": GraphQLArgument(GraphQLNonNull(GraphQLString))
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_create_conversation(info, **args)
                ),
                "sendMessage": GraphQLField(
                    ConversationType,
                    args={
                        "conversationId": GraphQLArgument(GraphQLNonNull(GraphQLString)),
                        "message": GraphQLArgument(GraphQLNonNull(MessageInput))
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_send_message(info, **args)
                ),
                "loadModule": GraphQLField(
                    ModuleType,
                    args={
                        "name": GraphQLArgument(GraphQLNonNull(GraphQLString))
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_load_module(info, **args)
                ),
                "unloadModule": GraphQLField(
                    GraphQLBoolean,
                    args={
                        "name": GraphQLArgument(GraphQLNonNull(GraphQLString))
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_unload_module(info, **args)
                ),
                "executeTool": GraphQLField(
                    GraphQLString,
                    args={
                        "name": GraphQLArgument(GraphQLNonNull(GraphQLString)),
                        "arguments": GraphQLArgument(GraphQLString)
                    },
                    resolve=lambda obj, info, **args: resolvers.resolve_execute_tool(info, **args)
                )
            }
        )
        
        return GraphQLSchema(
            query=query_type,
            mutation=mutation_type
        )
    
    
    class GraphQLAPI:
        """GraphQL API server."""
        
        def __init__(self):
            self.resolvers = GraphQLResolvers()
            self.schema = create_schema(self.resolvers)
        
        def execute(self, query: str, variables: Dict = None) -> Dict:
            """Execute GraphQL query."""
            result = graphql_sync(
                self.schema,
                query,
                variable_values=variables
            )
            
            response = {}
            
            if result.data:
                response["data"] = result.data
            
            if result.errors:
                response["errors"] = [
                    {"message": str(e)} for e in result.errors
                ]
            
            return response
        
        async def execute_async(self, query: str, variables: Dict = None) -> Dict:
            """Execute GraphQL query asynchronously."""
            result = await graphql(
                self.schema,
                query,
                variable_values=variables
            )
            
            response = {}
            
            if result.data:
                response["data"] = result.data
            
            if result.errors:
                response["errors"] = [
                    {"message": str(e)} for e in result.errors
                ]
            
            return response
        
        def create_flask_app(self) -> 'Flask':
            """Create Flask app with GraphQL endpoint."""
            if not HAS_FLASK:
                raise ImportError("Flask required for Flask GraphQL server")
            
            app = Flask(__name__)
            
            @app.route("/graphql", methods=["GET", "POST"])
            def graphql_endpoint():
                if request.method == "GET":
                    # GraphiQL interface
                    return GRAPHIQL_HTML
                
                data = request.get_json()
                query = data.get("query", "")
                variables = data.get("variables", {})
                
                result = self.execute(query, variables)
                return jsonify(result)
            
            return app
        
        def create_aiohttp_app(self) -> 'web.Application':
            """Create aiohttp app with GraphQL endpoint."""
            if not HAS_AIOHTTP:
                raise ImportError("aiohttp required for async GraphQL server")
            
            app = web.Application()
            
            async def graphql_handler(request):
                if request.method == "GET":
                    return web.Response(
                        text=GRAPHIQL_HTML,
                        content_type="text/html"
                    )
                
                data = await request.json()
                query = data.get("query", "")
                variables = data.get("variables", {})
                
                result = await self.execute_async(query, variables)
                return web.json_response(result)
            
            app.router.add_route("*", "/graphql", graphql_handler)
            
            return app
        
        def run(self, host: str = "0.0.0.0", port: int = 8000):
            """Run GraphQL server."""
            if HAS_FLASK:
                app = self.create_flask_app()
                app.run(host=host, port=port)
            elif HAS_AIOHTTP:
                app = self.create_aiohttp_app()
                web.run_app(app, host=host, port=port)
            else:
                raise ImportError("Flask or aiohttp required for GraphQL server")


else:
    class GraphQLAPI:
        def __init__(self):
            raise ImportError("graphql-core required for GraphQL API")


def create_graphql_api() -> GraphQLAPI:
    """Create GraphQL API instance."""
    if not HAS_GRAPHQL:
        raise ImportError("graphql-core required: pip install graphql-core")
    return GraphQLAPI()
