export function ScrollArea({ children, className }) {
  return <div className={`${className} overflow-y-auto`}>{children}</div>;
}
