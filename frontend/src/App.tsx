import WizardLayout from "./components/pages/wizardLayout";
import { CapabilitiesProvider } from "./models/capabilities";

function App() {
  return (
    <CapabilitiesProvider>
      <div>
        <WizardLayout />
      </div>
    </CapabilitiesProvider>
  );
}

export default App;
