import { useState, useCallback, useRef } from 'react'
import { uploadFile, type UploadResponse } from '@/lib/api'
import { Upload, FileText, Tag, Info, CheckCircle2, FileUp, Lightbulb, AlertCircle } from 'lucide-react'

export function UploadPanel() {
    const [domain, setDomain] = useState('')
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [isUploading, setIsUploading] = useState(false)
    const [result, setResult] = useState<UploadResponse | null>(null)
    const [error, setError] = useState<string | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const handleFileChange = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0]
            if (!file) return

            const fileName = file.name.toLowerCase()
            const isPdf = fileName.endsWith('.pdf')
            const isTxt = fileName.endsWith('.txt') || file.type.startsWith('text/')

            if (!isPdf && !isTxt) {
                setError('Only PDF and TXT files are supported')
                return
            }

            setSelectedFile(file)
            setError(null)
            setResult(null)

            // Auto-fill domain from filename if empty
            if (!domain) {
                const baseName = file.name.replace(/\.[^/.]+$/, '')
                setDomain(baseName.toUpperCase().replace(/[^A-Z0-9]/g, '_'))
            }
        },
        [domain]
    )

    const handleUpload = async () => {
        if (!selectedFile || !domain.trim()) return

        setIsUploading(true)
        setError(null)
        setResult(null)

        const { data, error: apiError } = await uploadFile(selectedFile, domain)

        if (apiError) {
            setError(apiError)
        } else {
            setResult(data)
            // Clear form on success
            if (data?.status === 'indexed') {
                setSelectedFile(null)
                setDomain('')
                if (fileInputRef.current) {
                    fileInputRef.current.value = ''
                }
            }
        }

        setIsUploading(false)
    }

    const isPdf = selectedFile?.name.toLowerCase().endsWith('.pdf')

    return (
        <div className="p-6 space-y-8">
            {/* Upload Section */}
            <div className="glass-card-inner p-8 stack-lg">
                <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center flex-shrink-0">
                        <FileUp className="icon-lg icon-accent" />
                    </div>
                    <div className="stack-xs">
                        <h3 className="text-high-contrast">Upload Materials</h3>
                        <p className="text-muted-subtle">
                            Add study content to enable AI-powered Q&A
                        </p>
                    </div>
                </div>

                {/* Helpful tip for first-time users */}
                <div className="bg-blue-500/5 border border-blue-500/15 rounded-xl p-4 flex gap-3">
                    <Lightbulb className="icon-md text-blue-400 flex-shrink-0" />
                    <div className="stack-xs">
                        <p className="text-xs text-blue-300/80 leading-relaxed">
                            Upload textbook chapters, lecture notes, or research papers (PDF or TXT).
                        </p>
                        <p className="text-xs text-muted-subtle">
                            PDFs are text-extracted only — images, diagrams, and scanned content are not processed.
                        </p>
                    </div>
                </div>

                <div className="stack-lg">
                    {/* File Upload */}
                    <div className="stack-sm">
                        <label className="text-label flex items-center gap-2">
                            <FileText className="icon-sm" />
                            Select File
                        </label>
                        <label className="input-recessed flex items-center gap-3 cursor-pointer hover:border-white/15 transition-colors">
                            <Upload className="icon-md icon-muted" />
                            <span className="text-low-contrast text-sm flex-1">
                                {selectedFile
                                    ? `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`
                                    : 'Choose a .pdf or .txt file...'
                                }
                            </span>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,.txt,text/plain,application/pdf"
                                onChange={handleFileChange}
                                disabled={isUploading}
                                className="hidden"
                            />
                            <span className="btn-secondary-quiet text-xs py-1.5 px-3">Browse</span>
                        </label>
                        {isPdf && (
                            <p className="text-xs text-muted-subtle flex items-center gap-1.5">
                                <Info className="icon-sm" />
                                PDF text will be extracted. Images & equations may not render.
                            </p>
                        )}
                    </div>

                    {/* Domain Tag */}
                    <div className="stack-sm">
                        <label className="text-label flex items-center gap-2">
                            <Tag className="icon-sm" />
                            Domain Tag
                        </label>
                        <input
                            type="text"
                            value={domain}
                            onChange={(e) => setDomain(e.target.value.toUpperCase())}
                            placeholder="e.g., PHYSICS, BIOLOGY"
                            disabled={isUploading}
                            className="input-recessed w-full"
                        />
                        <p className="text-muted-subtle text-xs">
                            Helps organize and retrieve content accurately
                        </p>
                    </div>

                    {/* Upload Button */}
                    <button
                        onClick={handleUpload}
                        disabled={isUploading || !selectedFile || !domain.trim()}
                        className="btn-primary-gradient w-full h-12 flex items-center justify-center gap-2"
                    >
                        <Upload className="icon-sm" />
                        {isUploading ? 'Processing...' : 'Upload & Index'}
                    </button>
                </div>
            </div>

            {/* Status Messages */}
            {error && (
                <div className="glass-card-inner p-5 border-amber-500/15 bg-amber-500/5 flex items-start gap-3">
                    <Info className="icon-md text-amber-400 flex-shrink-0" />
                    <div className="stack-xs">
                        <p className="text-sm text-amber-300/80">{error}</p>
                        <p className="text-xs text-muted-subtle">Check the file format and try again.</p>
                    </div>
                </div>
            )}

            {result && result.status === 'indexed' && (
                <div className="glass-card-inner p-5 border-green-500/20 bg-green-500/5 flex items-start gap-3">
                    <CheckCircle2 className="icon-md text-green-400 flex-shrink-0" />
                    <p className="text-sm text-green-300/80">
                        Indexed {result.word_count} words in {result.domain}.
                        Created {result.chunks_created} searchable chunks.
                    </p>
                </div>
            )}

            {result && result.status === 'rejected' && (
                <div className="glass-card-inner p-5 border-amber-500/15 bg-amber-500/5 flex items-start gap-3">
                    <AlertCircle className="icon-md text-amber-400 flex-shrink-0" />
                    <div className="stack-xs">
                        <p className="text-sm text-amber-300/80">
                            {result.rejection_reason}
                        </p>
                        <p className="text-xs text-muted-subtle">
                            Ensure file has at least 100 words of extractable text.
                        </p>
                    </div>
                </div>
            )}
        </div>
    )
}
