export default function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {Icon && <Icon className="w-10 h-10 text-border mb-4" strokeWidth={1.5} />}
      <h3 className="font-serif text-lg text-charcoal mb-2">{title}</h3>
      {description && <p className="text-muted text-sm max-w-md mb-6">{description}</p>}
      {action}
    </div>
  );
}
