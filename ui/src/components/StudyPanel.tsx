import { useState } from 'react'
import { generateStudyGuide, type StudyGuideResponse } from '@/lib/api'
import { Sparkles, BookOpen, Info, ChevronRight } from 'lucide-react'

type Difficulty = 'easy' | 'medium' | 'hard'

export function StudyPanel() {
    const [topic, setTopic] = useState('')
    const [difficulty, setDifficulty] = useState<Difficulty>('medium')
    const [isGenerating, setIsGenerating] = useState(false)
    const [result, setResult] = useState<StudyGuideResponse | null>(null)
    const [error, setError] = useState<string | null>(null)

    const handleGenerate = async () => {
        if (!topic.trim()) return

        setIsGenerating(true)
        setError(null)
        setResult(null)

        const { data, error: apiError } = await generateStudyGuide(topic, difficulty)

        if (apiError) {
            setError(apiError)
        } else {
            setResult(data)
        }

        setIsGenerating(false)
    }

    return (
        <div className="p-6 space-y-8">
            {/* Controls Section */}
            <div className="glass-card-inner p-8 stack-lg">
                <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center flex-shrink-0">
                        <BookOpen className="icon-lg icon-accent" />
                    </div>
                    <div className="stack-xs">
                        <h3 className="text-high-contrast">Generate Study Questions</h3>
                        <p className="text-muted-subtle">
                            Create practice questions from your materials
                        </p>
                    </div>
                </div>

                <div className="stack-lg">
                    {/* Topic Input */}
                    <div className="stack-sm">
                        <label className="text-label">Topic</label>
                        <input
                            type="text"
                            value={topic}
                            onChange={(e) => setTopic(e.target.value)}
                            placeholder="e.g., Thermodynamics, Neural Networks"
                            disabled={isGenerating}
                            className="input-recessed w-full"
                        />
                    </div>

                    {/* Difficulty Selection */}
                    <div className="stack-sm">
                        <label className="text-label">Difficulty</label>
                        <div className="flex gap-2">
                            {(['easy', 'medium', 'hard'] as Difficulty[]).map((level) => (
                                <button
                                    key={level}
                                    onClick={() => setDifficulty(level)}
                                    disabled={isGenerating}
                                    className={`flex-1 py-3 px-4 rounded-xl text-sm font-medium transition-all ${difficulty === level
                                        ? 'bg-gradient-to-br from-purple-500/25 to-blue-500/25 text-high-contrast border border-purple-500/30'
                                        : 'btn-secondary-quiet !rounded-xl'
                                        }`}
                                >
                                    {level === 'easy' && 'Easy'}
                                    {level === 'medium' && 'Medium'}
                                    {level === 'hard' && 'Hard'}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Generate Button */}
                    <button
                        onClick={handleGenerate}
                        disabled={isGenerating || !topic.trim()}
                        className="btn-primary-gradient w-full h-12 flex items-center justify-center gap-2"
                    >
                        <Sparkles className="icon-sm" />
                        {isGenerating ? 'Generating...' : 'Generate Questions'}
                    </button>
                </div>
            </div>

            {/* Error Display - calm styling */}
            {error && (
                <div className="glass-card-inner p-5 border-amber-500/15 bg-amber-500/5 flex items-start gap-3">
                    <Info className="icon-md text-amber-400 flex-shrink-0" />
                    <div className="stack-xs">
                        <p className="text-sm text-amber-300/80">{error}</p>
                        <p className="text-xs text-muted-subtle">Check your connection and try again.</p>
                    </div>
                </div>
            )}

            {/* Results */}
            {result && result.status === 'generated' && (
                <div className="glass-card-inner p-8 stack-lg">
                    <div className="flex items-center justify-between flex-wrap gap-3">
                        <h3 className="text-high-contrast">Generated Questions</h3>
                        <div className="flex gap-2">
                            <span className="pill-container text-xs font-medium bg-purple-500/15 text-purple-300 border border-purple-500/20">
                                {result.topic}
                            </span>
                            <span className="pill-container text-xs font-medium bg-white/5 text-low-contrast border border-white/8">
                                {result.difficulty}
                            </span>
                        </div>
                    </div>

                    <div className="stack-md">
                        {result.questions.map((q, index) => (
                            <div
                                key={index}
                                className="bg-white/3 rounded-xl p-5 flex gap-4 group hover:bg-white/5 transition-colors"
                            >
                                <div className="w-8 h-8 rounded-full bg-purple-500/10 flex items-center justify-center flex-shrink-0 text-sm font-semibold text-purple-400">
                                    {index + 1}
                                </div>
                                <div className="stack-xs flex-1">
                                    <p className="text-medium-contrast leading-relaxed">
                                        {q.question}
                                    </p>
                                    <p className="text-muted-subtle flex items-center gap-1">
                                        <ChevronRight className="icon-sm" />
                                        {q.expected_answer_length}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Failure Display - calm, informative */}
            {result && result.status === 'failed' && (
                <div className="glass-card-inner p-5 border-amber-500/15 bg-amber-500/5 flex items-start gap-3">
                    <Info className="icon-md text-amber-400 flex-shrink-0" />
                    <div className="stack-xs">
                        <p className="text-sm text-amber-300/80">
                            {result.failure_reason || 'Could not generate questions'}
                        </p>
                        <p className="text-xs text-muted-subtle">
                            Try a different topic or check if relevant materials are uploaded.
                        </p>
                    </div>
                </div>
            )}
        </div>
    )
}
