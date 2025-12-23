import { useState } from 'react'
import { askQuestion, type AskQuestionResponse } from '@/lib/api'
import { Send, MessageSquare, Info, CheckCircle2, Quote, HelpCircle, BookOpen } from 'lucide-react'

interface Message {
    id: string
    type: 'user' | 'assistant'
    content: string
    response?: AskQuestionResponse
}

export function ChatView() {
    const [messages, setMessages] = useState<Message[]>([])
    const [inputValue, setInputValue] = useState('')
    const [isLoading, setIsLoading] = useState(false)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!inputValue.trim() || isLoading) return

        const userMessage: Message = {
            id: Date.now().toString(),
            type: 'user',
            content: inputValue.trim(),
        }

        setMessages((prev) => [...prev, userMessage])
        setInputValue('')
        setIsLoading(true)

        const { data, error } = await askQuestion(userMessage.content)

        // Content comes from backend in structured format - no parsing needed
        const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: 'assistant',
            content: error
                ? `Error: ${error}`
                : data?.status === 'ANSWERED'
                    ? data.answer || ''
                    : data?.reason || 'Unable to process request',
            response: data || undefined,
        }

        setMessages((prev) => [...prev, assistantMessage])
        setIsLoading(false)
    }

    return (
        <div className="flex flex-col h-full">
            {/* Messages area */}
            <div className="flex-1 overflow-y-auto px-4 py-8 space-y-6">
                {/* Empty state with clear guidance */}
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center px-6">
                        <div className="glass-card-inner p-10 max-w-sm stack-lg">
                            <div className="w-14 h-14 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mx-auto">
                                <MessageSquare className="icon-lg icon-accent" />
                            </div>
                            <div className="stack-sm">
                                <h3 className="text-high-contrast">Ask a Question</h3>
                                <p className="text-medium-contrast text-sm leading-relaxed">
                                    Type a question about your uploaded study materials.
                                </p>
                            </div>
                            <div className="text-left bg-white/3 rounded-xl p-4 stack-xs">
                                <p className="text-muted-subtle text-xs font-medium uppercase tracking-wide">Try asking:</p>
                                <p className="text-low-contrast text-sm">"What is entropy in thermodynamics?"</p>
                                <p className="text-low-contrast text-sm">"Explain homeostasis in biology"</p>
                            </div>
                        </div>
                    </div>
                )}

                {messages.map((message) => {
                    const isAnswered = message.type === 'assistant' && message.response?.status === 'ANSWERED'

                    return (
                        <div
                            key={message.id}
                            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`max-w-[85%] px-5 py-4 ${message.type === 'user'
                                    ? 'bg-gradient-to-br from-purple-500/30 to-blue-500/30 rounded-2xl rounded-br-lg border border-purple-500/20'
                                    : 'glass-card-inner rounded-2xl rounded-bl-lg'
                                    }`}
                            >
                                {/* User message */}
                                {message.type === 'user' && (
                                    <p className="text-high-contrast whitespace-pre-wrap leading-relaxed">
                                        {message.content}
                                    </p>
                                )}

                                {/* Assistant message - answer is already clean from backend */}
                                {message.type === 'assistant' && (
                                    <p className={`whitespace-pre-wrap leading-relaxed ${isAnswered ? 'text-high-contrast text-base' : 'text-low-contrast'
                                        }`}>
                                        {message.content}
                                    </p>
                                )}

                                {/* Metadata for assistant messages */}
                                {message.type === 'assistant' && message.response && (
                                    <div className="mt-5 pt-4 border-t border-white/5 space-y-4">
                                        {/* Status badges */}
                                        <div className="flex gap-2 flex-wrap">
                                            {/* Answered status - show confidence */}
                                            {message.response.status === 'ANSWERED' && (
                                                <span className={`pill-container text-xs font-medium flex items-center gap-1.5 ${message.response.confidence === 'HIGH'
                                                    ? 'bg-green-500/10 text-green-400 border border-green-500/15'
                                                    : 'bg-blue-500/10 text-blue-400 border border-blue-500/15'
                                                    }`}>
                                                    {message.response.confidence === 'HIGH'
                                                        ? <CheckCircle2 className="icon-sm" />
                                                        : <Info className="icon-sm" />
                                                    }
                                                    {message.response.confidence === 'HIGH' ? '✓ Confident' : '⚠ Limited Confidence'}
                                                </span>
                                            )}

                                            {/* Refused status - calm, informative */}
                                            {message.response.status === 'REFUSED' && (
                                                <span className="pill-container text-xs font-medium flex items-center gap-1.5 bg-white/5 text-low-contrast border border-white/8">
                                                    <HelpCircle className="icon-sm" />
                                                    Cannot Answer
                                                </span>
                                            )}

                                            {/* Domain badge */}
                                            {message.response.domain && message.response.status === 'ANSWERED' && (
                                                <span className="pill-container text-xs font-medium bg-white/5 text-low-contrast border border-white/8 flex items-center gap-1.5">
                                                    <BookOpen className="icon-sm" />
                                                    {message.response.domain}
                                                </span>
                                            )}
                                        </div>

                                        {/* Quote - only if provided by backend */}
                                        {message.response.status === 'ANSWERED' && message.response.quote && (
                                            <div className="bg-white/3 rounded-xl p-4 flex gap-3">
                                                <Quote className="icon-md icon-muted flex-shrink-0 mt-0.5" />
                                                <p className="text-xs text-low-contrast italic leading-relaxed">
                                                    {message.response.quote}
                                                </p>
                                            </div>
                                        )}

                                        {/* Refusal reason - calm styling */}
                                        {message.response.status === 'REFUSED' && message.response.reason && (
                                            <div className="bg-white/3 rounded-xl p-4 flex gap-3">
                                                <Info className="icon-md icon-muted flex-shrink-0 mt-0.5" />
                                                <div className="stack-xs">
                                                    <p className="text-xs text-low-contrast leading-relaxed">
                                                        {message.response.reason}
                                                    </p>
                                                    <p className="text-xs text-muted-subtle">
                                                        Try rephrasing your question or check if relevant materials are uploaded.
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )
                })}

                {isLoading && (
                    <div className="flex justify-start">
                        <div className="glass-card-inner rounded-2xl rounded-bl-lg px-5 py-4">
                            <div className="flex items-center gap-2">
                                <div className="flex gap-1">
                                    <span className="w-2 h-2 bg-purple-400/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                    <span className="w-2 h-2 bg-purple-400/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                    <span className="w-2 h-2 bg-purple-400/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                </div>
                                <span className="text-muted-subtle text-sm">Analyzing materials...</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Input area */}
            <div className="p-5 pt-0">
                <form onSubmit={handleSubmit}>
                    <div className="flex gap-3 items-center bg-black/20 rounded-full p-1.5 pl-6 border border-white/8">
                        <input
                            type="text"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            placeholder="Ask about your study materials..."
                            disabled={isLoading}
                            className="flex-1 bg-transparent border-0 outline-none text-medium-contrast placeholder:text-low-contrast text-sm py-2"
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !inputValue.trim()}
                            className="btn-primary-gradient h-10 w-10 p-0 flex items-center justify-center !rounded-full"
                        >
                            <Send className="icon-sm" />
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
